from config import DATA_DIR, MODELS_DIR
import unsloth
import gc
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm
from collections import Counter
from typing import cast
import pandas as pd
import os

from src.utils.llm_as_a_judge import get_llm_judge_1v1
from src.author_classifier.retrieval_based_classification import predict_dataset_with_knn, train_retrieval_classifier
from src.evaluation.compare_content import get_introduction_from_paper, get_llm_classifier_prediction
from src.utils.llm import get_gpt_model, get_qwen_foundation_model

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'


def predict_perplexity(abstract, model, tokenizer):
    prefix = "Abstract:\n"

    full_text = prefix + abstract  # Maybe sinnvoll mit dem Titel des Papers? Genauso wird es ja auch in den Daten normalerweise sein + wir haben direkt ob das Abstract zum Titel aka zum Paper passt 🤔

    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
    prefix_length = prefix_ids.shape[1]

    input_ids = inputs.input_ids
    labels = input_ids.clone()
    labels[:, :prefix_length] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss  # TODO check for normalization

    # Perplexity = exp(NLL)
    perplexity = torch.exp(neg_log_likelihood)

    return perplexity.item()


def winrate_test(gpt_model, gpt_tokenizer, syn_abstract: str, ground_truth: str):
    winner_1, reasoning_1 = get_llm_judge_1v1(
        model=gpt_model,
        tokenizer=gpt_tokenizer,
        abstract_a=syn_abstract,
        abstract_b=ground_truth
    )

    if winner_1 == "A":
        score_1 = 1  # Gen gewinnt
    elif winner_1 == "B":
        score_1 = 0  # GT gewinnt
    else:
        score_1 = -1  # Fehler

    winner_2, reasoning_2 = get_llm_judge_1v1(
        model=gpt_model,
        tokenizer=gpt_tokenizer,
        abstract_a=ground_truth,
        abstract_b=syn_abstract
    )

    if winner_2 == "B":
        score_2 = 1  # Gen gewinnt
    elif winner_2 == "A":
        score_2 = 0  # GT gewinnt
    else:
        score_2 = -1  # Fehler

    if score_1 == -1 or score_2 == -1:
        final_win_score = -1
        final_reasoning = f"Error in Parsing. R1: {reasoning_1} | R2: {reasoning_2}"
    else:
        final_win_score = (score_1 + score_2) / 2.0
        final_reasoning = None

    return final_win_score, final_reasoning


def evaluate_abstracts(dataset_name: str, original_dataset_name: str, ground_truth_eval_file: str):
    device = "cuda"

    print(f"Lade Datensatz: {dataset_name}...")
    ds = Dataset.load_from_disk(f"{DATA_DIR}/abstract_predictions/{dataset_name}")
    if isinstance(ds, dict) and 'test' in ds:
        print("Datensatz hat einen 'test' split -> Evaluiere nur Test-Split")
        data = cast(Dataset, ds['test'])
    else:
        data = ds

    knn_dataset = DatasetDict.load_from_disk(
        f"{DATA_DIR}/datasets/{original_dataset_name}")
    knn_dataset = knn_dataset.class_encode_column('author')
    knn_dataset = knn_dataset.rename_column("author", "labels")
    if "ground_truth" in knn_dataset.column_names["train"]:
        knn_dataset = knn_dataset.rename_column("ground_truth", "abstract")

    knn_test_dataset = data.cast_column(
        "author", knn_dataset["train"].features["labels"])
    knn_test_dataset = knn_test_dataset.rename_column("author", "labels")

    # Author-Classifier (DistilBERT + KNN) laden
    model_path = f"{MODELS_DIR}/distilbert-base-uncased/model"
    print(f"Lade Classifier aus: {model_path}...")
    style_tokenizer = AutoTokenizer.from_pretrained(model_path)
    style_model = AutoModel.from_pretrained(model_path)
    style_model.to(device)
    style_model.eval()

    knn_train_dataset = knn_dataset["train"]
    if "val" in knn_dataset:
        knn_train_dataset = concatenate_datasets(
            [knn_train_dataset, knn_dataset["val"]])

    style_index, style_author_mapping = train_retrieval_classifier(
        knn_train_dataset, model_path, cosine_similarity=True)

    results = {}
    for col in ["gen_abstract", "gen_abstract_random_1", "gen_abstract_random_2"]:
        temp_test_ds = knn_test_dataset.rename_column(col, "abstract")

        predicted_author_ids, margin_ratios, mrrs = predict_dataset_with_knn(
            temp_test_ds,
            style_index,
            style_author_mapping,
            style_model,
            style_tokenizer,
            k=3,
            cosine_similarity=True
        )
        id2str = knn_dataset["test"].features['labels'].int2str
        results[col] = {
            "predicted_authors": [id2str(pid) for pid in predicted_author_ids],
            "margin_ratios": margin_ratios,
            "mrrs": mrrs
        }

    del style_model
    del style_tokenizer
    del style_index
    del style_author_mapping

    print("Lade Content-Evaluation Model (GPT)...")
    gpt_model, gpt_tokenizer = get_gpt_model()

    detailed_results = []

    pred_distribution = Counter()

    print("\nStarte Evaluation...")

    for i, row in enumerate(tqdm(data)):
        paper_id = row['paper_id']  # type: ignore
        true_author = row['author']  # type: ignore
        syn_abstract = row['gen_abstract']  # type: ignore

        # --- Style Evaluation (Synthetic) ---
        pred_author_syn = results["gen_abstract"]["predicted_authors"][i]
        pred_distribution[pred_author_syn] += 1

        # --- Content Evaluation ---
        is_content_same = 0
        error_msg = None

        try:
            introduction = get_introduction_from_paper(paper_id)
            if introduction:
                is_content_same, thinking = get_llm_classifier_prediction(
                    gpt_model, gpt_tokenizer, introduction, syn_abstract, verbose=True)

                if is_content_same == -1:
                    if thinking:
                        error_msg = thinking
                    else:
                        error_msg = "Probably max tokens reached. thinking couldn't be split."

            else:
                error_msg = "No Introduction found"
        except Exception as e:
            error_msg = str(e)
            print(f"Fehler bei Paper {paper_id}: {e}")

        # Winrate
        win_rate, reasoning = winrate_test(
            # type: ignore
            gpt_model, gpt_tokenizer, syn_abstract, row["ground_truth"])

        if reasoning:
            print("ERROR: Winrate wasn't successfull. Errors occured:")
            print(reasoning)

        detailed_results.append({
            "paper_id": paper_id,
            "true_author": true_author,
            "content_match_syn": is_content_same,
            "win_rate": win_rate,

            # "Normal" Summary
            "predicted_author_syn": results["gen_abstract"]["predicted_authors"][i],
            "margin_ratio_syn": results["gen_abstract"]["margin_ratios"][i],
            "mrr_syn": results["gen_abstract"]["mrrs"][i],

            # Random Summary 1
            "random_summary_1_id": row['random_summary_1_id'],  # type: ignore
            "predicted_author_rndm_1": results["gen_abstract_random_1"]["predicted_authors"][i],
            "margin_ratio_rndm_1": results["gen_abstract_random_1"]["margin_ratios"][i],
            "mrr_rndm_1": results["gen_abstract_random_1"]["mrrs"][i],

            # Random Summary 2
            "random_summary_2_id": row['random_summary_2_id'],  # type: ignore
            "predicted_author_rndm_2": results["gen_abstract_random_2"]["predicted_authors"][i],
            "margin_ratio_rndm_2": results["gen_abstract_random_2"]["margin_ratios"][i],
            "mrr_rndm_2": results["gen_abstract_random_2"]["mrrs"][i],

            # Rest
            "error_log": error_msg,
            "gen_abstract": syn_abstract
        })

    # Perplexity evaluierung:
    print("Entlade GPT und Style Modell für VRAM...")
    del gpt_model
    del gpt_tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    print("Starte Perplexity evaluierung.")
    print("Lade Foundation Modell...")
    ppl_model, ppl_tokenizer = get_qwen_foundation_model()

    print("Berechne Perplexity für alle Abstracts...")
    for result in tqdm(detailed_results):
        abstract = result['gen_abstract']  # type: ignore
        result["perplexity"] = predict_perplexity(
            abstract, ppl_model, ppl_tokenizer)

    parent_dir = f"{DATA_DIR}/abstract_evaluation/{dataset_name}"
    os.makedirs(parent_dir, exist_ok=True)

    print("\nSpeichere detaillierte Ergebnisse in 'abstract_evaluation_details.csv'...")
    df_results = pd.DataFrame(detailed_results)

    # --- Ground Truth Daten laden und mergen ---
    print(
        f"Lade Ground Truth Evaluation aus '{ground_truth_eval_file}.csv'...")
    gt_file_path = f"{DATA_DIR}/abstract_evaluation/{ground_truth_eval_file}.csv"

    if not os.path.exists(gt_file_path):
        print(f"Fehler: Ground Truth Datei {gt_file_path} nicht gefunden!")
        return

    df_gt = pd.read_csv(gt_file_path, dtype={'paper_id': str})

    df_results['paper_id'] = df_results['paper_id'].astype(str)

    df_gt_subset = df_gt[['paper_id', 'predicted_author_gt', 'content_match_gt', 'perplexity', 'mrr_gt', 'margin_ratio_gt']].rename(
        columns={'perplexity': 'perplexity_ground_truth'}
    )

    # Merge basierend auf der paper_id
    df_results = pd.merge(df_results, df_gt_subset, on='paper_id', how='left')
    if df_results['perplexity_ground_truth'].isna().any():
        print("Warnung: Nicht für alle Paper-IDs konnte ein Ground Truth Wert gefunden werden.")

    df_export = df_results.drop(columns=['gen_abstract', "predicted_author_gt",
                                "content_match_gt", "perplexity_ground_truth", "margin_ratio_gt", "mrr_gt"])
    df_export.to_csv(
        f"{parent_dir}/abstract_evaluation_details.csv", index=False)
    print("Speichern erfolgreich.")

    df_dist = pd.DataFrame.from_dict(
        pred_distribution, orient='index', columns=['count'])
    df_dist_sorted = df_dist.sort_values(by='count', ascending=False)
    df_dist_sorted.to_csv(f"{parent_dir}/author_prediction_distribution.csv")
    print("Verteilung der vorhergesagten Autoren in 'author_prediction_distribution.csv' gespeichert.")

    create_report(df_results, df_dist, parent_dir)


def create_report(df: pd.DataFrame, df_dist: pd.DataFrame, parent_folder: str):
    df_dist_sorted = df_dist.sort_values(by='count', ascending=False)

    # Sicherstellen, dass alle IDs Strings sind (wichtig für das Mapping gleich)
    df["paper_id"] = df["paper_id"].astype(str)
    if 'random_summary_1_id' in df.columns:
        df["random_summary_1_id"] = df["random_summary_1_id"].astype(str)
        df["random_summary_2_id"] = df["random_summary_2_id"].astype(str)

    total = len(df)

    # --- Content Metrics ---
    acc_content = df.loc[df['content_match_syn']
                         != -1, 'content_match_syn'].mean()
    acc_content_gt = df.loc[df['content_match_gt']
                            != -1, 'content_match_gt'].mean()
    avg_win_rate = df.loc[df['win_rate'] != -1, 'win_rate'].mean()

    # --- Style Accuracy (Voting) ---
    acc_style_syn_test = (df['predicted_author_syn'] ==
                          df['true_author']).sum() / len(df)
    acc_style_gt_test = (df['predicted_author_gt'] ==
                         df['true_author']).sum() / len(df)
    acc_style_syn_gt = (df['predicted_author_syn'] ==
                        df['predicted_author_gt']).sum() / len(df)

    acc_style_random = ((df['predicted_author_rndm_1'] == df['true_author']).sum(
    ) + (df['predicted_author_rndm_2'] == df['true_author']).sum()) / (len(df)*2)

    # MRR Averages
    avg_mrr_syn = df['mrr_syn'].mean()
    avg_mrr_rndm = pd.concat([df['mrr_rndm_1'], df['mrr_rndm_2']]).mean()
    avg_mrr_gt = df['mrr_gt'].mean()

    # Margin Ratio Averages (inf Werte filtern, falls vorhanden)
    avg_margin_syn = df['margin_ratio_syn'].replace(
        [np.inf, -np.inf], np.nan).dropna().mean()
    avg_margin_rndm = pd.concat([df['margin_ratio_rndm_1'], df['margin_ratio_rndm_2']]).replace(
        [np.inf, -np.inf], np.nan).dropna().mean()
    avg_margin_gt = df['margin_ratio_gt'].replace(
        [np.inf, -np.inf], np.nan).dropna().mean()

    # --- Random Author Difference ---
    author_mapping = dict(zip(df['paper_id'], df['true_author']))
    true_author_rndm_1 = df['random_summary_1_id'].map(author_mapping)
    true_author_rndm_2 = df['random_summary_2_id'].map(author_mapping)

    diff_rndm_1 = (df['predicted_author_rndm_1'] != true_author_rndm_1).sum()
    diff_rndm_2 = (df['predicted_author_rndm_2'] != true_author_rndm_2).sum()
    acc_author_diff_random = (diff_rndm_1 + diff_rndm_2) / (total * 2)

    # --- Perplexity ---
    avg_perplexity = df['perplexity'].mean()
    median_perplexity = df['perplexity'].median()

    avg_perplexity_gt = df['perplexity_ground_truth'].mean()
    median_perplexity_gt = df['perplexity_ground_truth'].median()

    avg_diff_perplexity = (
        df['perplexity'] - df['perplexity_ground_truth']).abs().dropna().mean()

    # --- Report Generierung ---
    lines = []
    lines.append("-" * 60)
    lines.append(f"EVALUATIONS REPORT")
    lines.append("-" * 60)
    lines.append(f"Dataset evaluated: {parent_folder.split('/')[-1]}")
    lines.append(f"Total Samples: {total}")
    lines.append(f"Test Samples: {len(df)}")
    lines.append("-" * 30)

    lines.append("METRICS:")
    lines.append(
        f"  Content Consistency Score:     {acc_content:.2%} (GT: {acc_content_gt:.2%})")
    lines.append(f"  Win Rate against ground truth: {avg_win_rate:.2%}")
    lines.append("")
    lines.append(
        f"  Average Perplexity (PPL):      {avg_perplexity:.2f} (GT: {avg_perplexity_gt:.2f})")
    lines.append(
        f"  Median Perplexity (PPL):       {median_perplexity:.2f} (GT: {median_perplexity_gt:.2f})")
    lines.append("")
    lines.append(f"  Average Perplexity Diff:       {avg_diff_perplexity:.2f}")
    lines.append("")

    lines.append("STYLE CLASSIFICATION (Test Dataset):")
    lines.append(
        f"  Style Accuracy (Syn Test):     {acc_style_syn_test:.2%} (GT: {acc_style_gt_test:.2%})")
    lines.append(
        f"  Style Accuracy (Random):       {acc_style_random:.2%} (Complete random guessing: {1/len(df['true_author'].unique()):.2%})")
    lines.append(f"  Style Accuracy (Syn = GT):     {acc_style_syn_gt:.2%}")
    lines.append(
        f"  Random Author Diff Rate:       {acc_author_diff_random:.2%} (Pred. author differs from orig. summary)")
    lines.append("")

    lines.append("CONTINUOUS STYLE METRICS:")
    lines.append(
        f"  Avg Margin Ratio (Syn Test):   {avg_margin_syn:.4f} (GT: {avg_margin_gt:.2f}) (< 1 -> correct author is closest)")
    lines.append(f"  Avg Margin Ratio (Random):     {avg_margin_rndm:.4f}")
    lines.append(
        f"  Average MRR (Syn Test):        {avg_mrr_syn:.4f} (GT: {avg_mrr_gt:.2f}) (Higher = Better)")
    lines.append(f"  Average MRR (Random):          {avg_mrr_rndm:.4f}")

    lines.append("-" * 30)
    lines.append("For Obsidian:")
    lines.append(
        f"| {acc_content:.2%} | {avg_perplexity:.2f} | {avg_diff_perplexity:.2f} | {avg_win_rate:.2%} |")
    lines.append(
        f"| {acc_style_syn_test:.2%} | {acc_style_random:.2%} | {avg_margin_syn:.2f} | {avg_margin_rndm:.2f} | {avg_mrr_syn:.2f} | {avg_mrr_rndm:.2f} |")
    lines.append("-" * 30)

    lines.append("TOP 5 PREDICTED AUTHORS (SYNTHETIC):")
    top_5_str = df_dist_sorted.head(5).to_string()
    lines.append(top_5_str)
    lines.append("-" * 60)

    if 'perplexity' in df.columns:
        lines.append("\nTOP 3 ABSTRACTS HIGHEST PERPLEXITY:")
        highest_3 = df.sort_values(by='perplexity', ascending=False).head(3)

        for i, (idx, row) in enumerate(highest_3.iterrows()):
            lines.append(
                f"\n{i+1}. Paper ID: {row['paper_id']} | PPL: {row['perplexity']:.2f}")
            text_snippet = row['gen_abstract']
            lines.append(f"   Abstract:\n {text_snippet}")

        lines.append("")
        lines.append("-" * 30)
        lines.append("\nTOP 3 ABSTRACTS LOWEST PERPLEXITY")
        lowest_3 = df.sort_values(by='perplexity', ascending=True).head(3)

        for i, (idx, row) in enumerate(lowest_3.iterrows()):
            lines.append(
                f"\n{i+1}. Paper ID: {row['paper_id']} | PPL: {row['perplexity']:.2f}")
            text_snippet = row['gen_abstract']
            lines.append(f"   Abstract:\n {text_snippet}")

    summary_text = "\n".join(lines)

    print(summary_text)

    with open(f"{parent_folder}/evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nZusammenfassung wurde in 'evaluation_summary.txt' gespeichert.")


def evaluate_ground_truth(dataset_name: str):
    device = "cuda"

    print(f"\nStarte Evaluation für Ground Truth: {dataset_name}...")

    dataset_path = f"{DATA_DIR}/datasets/{dataset_name}"
    print(f"Lade Test-Daten aus: {dataset_path}")
    loaded_data = DatasetDict.load_from_disk(dataset_path)

    if "test" in loaded_data:
        dataset = loaded_data["test"]
    else:
        print(f"Split 'test' nicht gefunden.")
        return

    knn_dataset = loaded_data.class_encode_column('author')
    knn_dataset = knn_dataset.rename_column("author", "labels")

    # 1. Author-Classifier (DistilBERT + KNN) laden
    model_path = f"{MODELS_DIR}/distilbert-base-uncased/model"
    print(f"Lade Classifier aus: {model_path}...")
    style_tokenizer = AutoTokenizer.from_pretrained(model_path)
    style_model = AutoModel.from_pretrained(model_path)
    style_model.to(device)
    style_model.eval()

    knn_train_dataset = knn_dataset["train"]
    if "val" in knn_dataset:
        knn_train_dataset = concatenate_datasets(
            [knn_train_dataset, knn_dataset["val"]])

    style_index, style_author_mapping = train_retrieval_classifier(
        knn_train_dataset, model_path, cosine_similarity=True)

    # --- Anpassung 1: Unpacking des Triples ---
    predicted_author_ids, margin_ratios_gt, mrrs_gt = predict_dataset_with_knn(
        knn_dataset["test"],
        style_index,
        style_author_mapping,
        style_model,
        style_tokenizer,
        k=3,
        cosine_similarity=True
    )

    id2str = knn_dataset["test"].features['labels'].int2str
    predicted_authors = [id2str(pid) for pid in predicted_author_ids]

    # 2. Content-Evaluation Model (GPT) laden
    print("Lade Content-Evaluation Model (GPT)...")
    gpt_model, gpt_tokenizer = get_gpt_model()

    detailed_results = []

    print("\nEvaluiere Style und Content für Ground Truth Abstracts...")
    for i, row in enumerate(tqdm(dataset)):
        paper_id = row["paper_id"]  # type: ignore
        gt_abstract = row['abstract']  # type: ignore
        true_author = row['author']  # type: ignore

        # --- Style Evaluation ---
        pred_author_gt = predicted_authors[i]

        # --- Anpassung 2: MRR und Margin Ratio für aktuellen Loop-Durchlauf extrahieren ---
        margin_ratio_gt = margin_ratios_gt[i]
        mrr_gt = mrrs_gt[i]

        # --- Content Evaluation ---
        is_content_same = 0
        error_msg = None

        try:
            introduction = get_introduction_from_paper(paper_id)
            if introduction:
                # Hier evaluieren wir die Introduction gegen den GT Abstract
                is_content_same, thinking = get_llm_classifier_prediction(
                    gpt_model, gpt_tokenizer, introduction, gt_abstract, verbose=True
                )

                if is_content_same == -1:
                    if thinking:
                        error_msg = thinking
                    else:
                        error_msg = "Probably max tokens reached. thinking couldn't be split."
            else:
                error_msg = "No Introduction found"
        except Exception as e:
            error_msg = str(e)
            print(f"Fehler bei Paper {paper_id}: {e}")

        # --- Anpassung 3: Metriken in die detailed_results aufnehmen ---
        detailed_results.append({
            "paper_id": paper_id,
            "true_author": true_author,
            "predicted_author_gt": pred_author_gt,
            "margin_ratio_gt": margin_ratio_gt,
            "mrr_gt": mrr_gt,
            "content_match_gt": is_content_same,
            "error_log": error_msg,
            "abstract": gt_abstract  # Zwischenspeichern für Perplexity
        })

    # --- Speicher freigeben ---
    print("\nEntlade GPT und Style Modell für VRAM...")
    del gpt_model
    del gpt_tokenizer
    del style_model
    del style_tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    # 3. Perplexity Modell (Qwen) laden
    print("Starte Perplexity evaluierung.")
    print("Lade Foundation Modell...")
    ppl_model, ppl_tokenizer = get_qwen_foundation_model()

    print("Berechne Perplexity für alle GT Abstracts...")
    for result in tqdm(detailed_results):
        abstract = result['abstract']
        result["perplexity"] = predict_perplexity(
            abstract, ppl_model, ppl_tokenizer)

    print("\nSpeichere Ergebnisse in 'ground_truth_evaluation.csv'...")
    df_results = pd.DataFrame(detailed_results)

    # Den Abstract-Text droppen, um das CSV kompakt zu halten
    if 'abstract' in df_results.columns:
        df_export = df_results.drop(columns=['abstract'])
    else:
        df_export = df_results

    os.makedirs(f"{DATA_DIR}/abstract_evaluation", exist_ok=True)
    df_export.to_csv(
        f"{DATA_DIR}/abstract_evaluation/{dataset_name}_gt_evaluation.csv", index=False)
    print("Speichern erfolgreich.")


if __name__ == "__main__":
    evaluate_abstracts("Qwen3_lora_3_lay_hyper_steering_abstracts_dataset",
                       "paper_dataset_icl", "paper_dataset_gt_evaluation")
