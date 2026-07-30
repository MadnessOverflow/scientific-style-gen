from time import time

from config import DATA_DIR, MODELS_DIR
import ast
import logging
import textwrap
from typing import List, Dict, Any, Tuple, cast
from datasets import Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm
import random
import pandas as pd

from src.utils.llm import generate_text, get_gpt_model, split_thinking_output
from src.utils.papers import download_paper_src, get_paper_full_tex, get_introduction, clean_latex
from src.utils.prompts import CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_USER_PROMPT


def get_ids_from_dataset(dataset_path: str):
    df = pd.read_csv(dataset_path)

    df['paper_list'] = df['paper_list'].apply(ast.literal_eval)

    author_dict = df.set_index('author')['paper_list'].to_dict()

    return author_dict


def get_introduction_from_paper(arxiv_id: str):
    path = download_paper_src(arxiv_id, f"{DATA_DIR}/papers")
    if not path:
        return None

    paper_tex = get_paper_full_tex(path)
    if not paper_tex:
        return None

    introduction = get_introduction(paper_tex)
    if introduction:
        introduction = clean_latex(introduction)

    return introduction


def create_dataset():
    random.seed(67)

    print("Starte erstellen von Datenset.")
    dataset_dict = DatasetDict.load_from_disk(f"{DATA_DIR}/datasets/paper_dataset")
    all_splits = list(dataset_dict.values())
    dataset = concatenate_datasets(all_splits)

    all_ids = sorted(list(dataset['paper_id']))

    data_entries = []

    for row in tqdm(dataset):
        paper_id = row['paper_id']  # type: ignore
        abstract = row['abstract']  # type: ignore

        pos_introduction = get_introduction_from_paper(paper_id)

        random_id = random.choice(all_ids)
        while random_id == paper_id:
            random_id = random.choice(all_ids)

        neg_introduction = get_introduction_from_paper(random_id)

        if not abstract:
            print(f"Abstract not found for paper: {paper_id}")
            continue
        if not pos_introduction:
            print(f"Introduction not found for paper: {paper_id}")
            continue
        if not neg_introduction:
            print(f"Introduction not found for paper: {random_id}")
            continue

        data_entries.append({
            "abstract": abstract,
            "introduction": pos_introduction,
            "label": True
        })
        data_entries.append({
            "abstract": abstract,
            "introduction": neg_introduction,
            "label": False
        })

    hf_dataset = Dataset.from_list(data_entries)
    hf_dataset.save_to_disk(f"{DATA_DIR}/datasets/content_classification_dataset")

    print(
        f"Dataset gespeichert. Gesamtgröße: {len(hf_dataset)} (50% Positiv, 50% Negativ).")
    return hf_dataset


def get_llm_classifier_prediction(model, tokenizer, introduction: str, abstract: str, max_prompt_tokens=3500, verbose=False) -> Tuple[int, str | None]:
    """
    Erstellt den Prompt und kürzt gezielt die Introduction, falls das 
    Gesamtlimit von Tokens überschritten wird.
    """
    # Berechnen der übrigen Tokens für die Introduction
    base_prompt_structure = CLASSIFIER_USER_PROMPT.format(
        introduction="", abstract=abstract)
    base_ids = tokenizer.encode(
        base_prompt_structure, add_special_tokens=False)
    sys_ids = tokenizer.encode(
        CLASSIFIER_SYSTEM_PROMPT, add_special_tokens=False)

    used_tokens = len(base_ids) + len(sys_ids)
    available_tokens_for_intro = max_prompt_tokens - used_tokens

    intro_ids = tokenizer.encode(introduction, add_special_tokens=False)

    if len(intro_ids) > available_tokens_for_intro:
        # Kürzen der Introduction, weil zu viele Tokens
        print(
            f"Introduction ist zu lang und wird gekürzt, damit der gesammte Prompt {max_prompt_tokens} Tokens hat.")
        trimmed_intro_ids = intro_ids[:max(0, available_tokens_for_intro - 5)]
        introduction = tokenizer.decode(
            trimmed_intro_ids, skip_special_tokens=True)

    prompt = CLASSIFIER_USER_PROMPT.format(
        introduction=introduction, abstract=abstract)
    if verbose:
        print(f"Prompt:\n{prompt}")

    response_text = generate_text(
        model, tokenizer, CLASSIFIER_SYSTEM_PROMPT, prompt, max_tokens=1024, do_sample=False)

    response, thinking_analysis = split_thinking_output(response_text)
    print(f"\nResponse: {response}")
    print(f"Model thinking process: {thinking_analysis}\n")

    clean_response = response.strip().lower().replace(".", "")

    if "true" in clean_response:
        return 1, thinking_analysis
    elif "false" in clean_response:
        return 0, thinking_analysis
    elif "error" in clean_response:
        return -1, thinking_analysis
    else:
        # Fallback falls das LLM scheiße labert
        return -1, thinking_analysis


def test_llm_as_topic_classifier(model, tokenizer):
    model_name = model.config.name_or_path

    dataset_path = f"{DATA_DIR}/datasets/content_classification_dataset"

    print(f"Lade Dataset von {dataset_path}...")
    eval_data = Dataset.load_from_disk(dataset_path)

    results_log: List[Dict[str, Any]] = []

    print(f"Starte Evaluierung mit Modell: {model_name}")

    try:
        for sample in tqdm(eval_data):
            sample_dict = cast(Dict[str, Any], sample)

            intro = sample_dict.get('introduction', "")
            abstract = sample_dict.get('abstract', "")
            target_label = 1 if sample_dict.get('label', False) else 0

            predicted_label, thinking_text = get_llm_classifier_prediction(
                model, tokenizer, intro, abstract)

            results_log.append({
                "introduction": intro,
                "abstract": abstract,
                "target": target_label,
                "prediction": predicted_label,
                "is_correct": predicted_label == target_label,
                "model_reasoning": thinking_text if thinking_text else ""
            })
    except Exception as e:
        print(e)

    df = pd.DataFrame(results_log)

    accuracy = df[df['prediction'] != -1]["is_correct"].mean()
    print(f"\n--- Ergebnis ---")
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")

    output_filename = f"{dataset_path}/results_{model_name.replace('/', '_')}.csv"

    df.to_csv(output_filename, index=False)

    print(f"Ergebnisse gespeichert in: {output_filename}")


def print_wrong_examples(model_name: str):
    log_filename = f'{DATA_DIR}/datasets/content_classification_dataset/evaluation_{model_name.replace("/", "_")}.log'

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(message)s',
        filemode='w',
        encoding='utf-8',
        force=True
    )

    print(f"Schreibe Logs in Datei: {log_filename} ...")

    df_all = pd.read_csv(
        f"{DATA_DIR}/datasets/content_classification_dataset/results_{model_name.replace('/', '_')}.csv")
    df = df_all[df_all["is_correct"] == False]

    logging.info("=" * 80)
    logging.info("Allgemeine Statistiken:")
    logging.info("-" * 80)

    fn = df[(df['target'] == 1) & (df['prediction'] == 0)]
    fp = df[(df['target'] == 0) & (df['prediction'] == 1)]

    tp = df_all[(df_all['target'] == 1) & (df_all['prediction'] == 1)]
    tn = df_all[(df_all['target'] == 0) & (df_all['prediction'] == 0)]

    logging.info(f"Falsche Vorhersagen: {len(df)}")
    logging.info(f"False Negatives: {len(fn)}")
    logging.info(f"False Positives: {len(fp)}")
    logging.info(f"True Positives: {len(tp)}")
    logging.info(f"True Negatives: {len(tn)}")
    logging.info(f"Anzahl an Samples: {len(df_all)}")

    logging.info("\n\n")

    for index, row in df.iterrows():
        target = row['target']
        pred = row['prediction']

        if target == 1 and pred == 0:
            error_type = "FALSE NEGATIVE (Modell dachte: Ungleich | War aber: Gleich)"
        elif target == 0 and pred == 1:
            error_type = "FALSE POSITIVE (Modell dachte: Gleich | War aber: Ungleich)"
        else:
            error_type = "Unbekannt"

        logging.info("=" * 80)
        logging.info(f"DATENSATZ INDEX: {index}")
        logging.info(f"Target: {target} | Prediction: {pred}")
        logging.info(f"Fehler-Typ: {error_type}")
        logging.info("-" * 80)

        logging.info(">>> ABSTRACT:")
        logging.info(textwrap.fill(str(row['abstract']), width=80))
        logging.info("")

        logging.info(">>> INTRODUCTION:")
        logging.info(textwrap.fill(str(row['introduction']), width=80))
        logging.info("\n\n")

    print("Fertig! Die Datei kann nun geöffnet werden.")


if __name__ == "__main__":
    create_dataset()

    model, tokenizer = get_gpt_model()
    start_time = time()
    test_llm_as_topic_classifier(model, tokenizer)
    print(f"Evaluierung abgeschlossen in {int((time() - start_time) // 60)} Minuten.")

    print_wrong_examples("unsloth/gpt-oss-20b-unsloth-bnb-4bit")
