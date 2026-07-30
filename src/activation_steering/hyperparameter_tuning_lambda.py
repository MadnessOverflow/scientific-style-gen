import unsloth
from config import DATA_DIR, MODELS_DIR
import re
import statistics
import time
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import DatasetDict, Dataset
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file

# Importiere deine bestehenden Hilfsfunktionen
from src.inference.prediction_activation_steering import get_steering_hook
from src.evaluation.evaluate_abstracts import predict_perplexity
from src.utils.prompts import DEFAULT_ABSTRACT_SYSTEM_PROMPT, DEFAULT_ABSTRACT_USER_PROMPT
from src.utils.llm import get_qwen_model, get_qwen_foundation_model
from src.author_classifier.retrieval_based_classification import predict_dataset_with_knn, train_retrieval_classifier


def calculate_ground_truth_ce_loss(prompt: str, ground_truth_abstract: str, model, tokenizer):
    """
    Berechnet den Cross-Entropy Loss für das Ground-Truth-Abstract, 
    gegeben den Prompt (Teacher Forcing).
    """
    # Das echte Abstract an den Prompt anhängen (plus EOS Token)
    full_text = prompt + ground_truth_abstract + tokenizer.eos_token

    # Tokenisieren
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids

    prompt_length = prompt_ids.shape[1]

    # Labels vorbereiten (Kopie der Input IDs)
    labels = full_ids.clone()

    # Wir wollen den Loss NUR für das Abstract berechnen, nicht für den Prompt.
    # Daher maskieren wir alle Prompt-Tokens mit -100 (Standard-Ignore-Index in PyTorch).
    labels[:, :prompt_length] = -100

    # Forward Pass (Modell berechnet intern den CE-Loss, wenn Labels übergeben werden)
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids.to(model.device),
            labels=labels.to(model.device)
        )
        ce_loss = outputs.loss

    return ce_loss.item()


def run_tuning_pipeline(
    steering_vectors_path: str,
    original_dataset_name: str,
    target_layer_indices: list[int],
    alphas_to_test: list[float]
):
    dataset_path = f"{DATA_DIR}/datasets/{original_dataset_name}"

    # 1. Daten laden (Validation Split!)
    print(f"Lade Daten aus: {dataset_path}")
    loaded_data = DatasetDict.load_from_disk(dataset_path)

    if "val" in loaded_data:
        val_dataset = loaded_data["val"]
    elif "validation" in loaded_data:
        val_dataset = loaded_data["validation"]
    else:
        raise ValueError(
            "Kein Validation-Split gefunden! Bitte erstelle einen 'val' Split.")

    print(f"Nutze Validation Split mit {len(val_dataset)} Samples.")

    print("Trainiere Style-Classifier für Evaluation...")
    model_path = f"{MODELS_DIR}/distilbert-base-uncased/model"
    style_tokenizer = AutoTokenizer.from_pretrained(model_path)
    style_model = AutoModel.from_pretrained(model_path).to("cuda").eval()

    knn_dataset = loaded_data.class_encode_column('author').rename_column(
        "author", "labels").rename_column("ground_truth", "abstract")
    style_index, style_author_mapping = train_retrieval_classifier(
        knn_dataset["train"], model_path, cosine_similarity=True)

    # 3. Modelle für Generierung und Perplexity laden
    print("Lade Modelle...")
    gen_model, gen_tokenizer = get_qwen_model()
    ppl_model, ppl_tokenizer = get_qwen_foundation_model()
    steering_vectors = load_file(steering_vectors_path)

    def format_inference_prompts(example):
        messages = [
            {"role": "system", "content": DEFAULT_ABSTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": DEFAULT_ABSTRACT_USER_PROMPT.format(
                summary=example["summary"])},
        ]

        result = gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {"text_prompt": result}

    val_dataset = val_dataset.map(format_inference_prompts)

    # 4. Resultate speichern
    tuning_results = []

    # 5. Iteration über die Alphas
    for alpha in alphas_to_test:
        print("\n" + "="*50)
        print(f"STARTE RUN FÜR LAMBDA (ALPHA) = {alpha}")
        print("="*50)

        start_time = time.time()

        # --- A. GENERIERUNG (vereinfachte Version deiner Funktion) ---
        # (Hier müsstest du deine Prompts wie gewohnt formatieren, ich halte es kurz zur Übersicht)
        # HINWEIS: Füge hier deine Prompt-Formatierung (format_inference_prompts) ein.

        generated_abstracts = []
        true_authors = []
        paper_ids = []
        cross_entropy_loss = []

        for row in tqdm(val_dataset, desc=f"Generiere (Alpha={alpha})"):
            paper_id = row["paper_id"]  # type: ignore
            prompt = row["text_prompt"]  # type: ignore

            paper_layer_vectors = {}

            for layer_idx in target_layer_indices:
                dict_key = f"{paper_id}_layer_{layer_idx}"
                if dict_key in steering_vectors:
                    paper_layer_vectors[layer_idx] = steering_vectors[dict_key]
                else:
                    break

            inputs = gen_tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=4096
            ).to(gen_model.device)

            input_ids = inputs["input_ids"]

            active_hooks = []
            for layer_idx in target_layer_indices:
                target_layer = gen_model.model.layers[layer_idx]
                vector_for_layer = paper_layer_vectors[layer_idx]

                hook_handle = target_layer.register_forward_hook(
                    get_steering_hook(vector_for_layer, alpha=alpha))
                active_hooks.append(hook_handle)

            try:
                with torch.inference_mode():
                    outputs = gen_model.generate(
                        input_ids=input_ids,
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=1024,
                        do_sample=True,
                        use_cache=True,
                        temperature=0.7,
                        top_p=0.8,
                        top_k=20,
                        min_p=0,
                        repetition_penalty=1.05,
                        eos_token_id=gen_tokenizer.eos_token_id,
                    )

                ground_truth_text = row["ground_truth"]  # type: ignore
                cross_entropy_loss.append(calculate_ground_truth_ce_loss(
                    prompt, ground_truth_text, gen_model, gen_tokenizer))
            finally:
                for hook in active_hooks:
                    hook.remove()

            input_length = input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]

            decoded_text = gen_tokenizer.decode(
                generated_tokens, skip_special_tokens=True)

            # Cleanup
            cleaned_text = re.sub(r"\*\*abstract\*\*\s*",
                                  "", decoded_text, flags=re.IGNORECASE).strip()

            generated_abstracts.append(cleaned_text)
            true_authors.append(row["author"])  # type: ignore
            paper_ids.append(paper_id)

        # --- B. EVALUIERUNG ---
        print(f"Evaluiere Alpha = {alpha}...")

        # Temporäres Dataset für den Classifier
        temp_eval_dict = {"paper_id": paper_ids, "abstract": generated_abstracts, "labels": [
            knn_dataset["train"].features['labels'].str2int(a) for a in true_authors]}
        temp_eval_ds = Dataset.from_dict(temp_eval_dict)

        # Style Predict
        predicted_author_ids, _, _ = predict_dataset_with_knn(
            temp_eval_ds, style_index, style_author_mapping, style_model, style_tokenizer, k=3, cosine_similarity=True
        )

        id2str = knn_dataset["train"].features['labels'].int2str
        predicted_authors = [id2str(pid) for pid in predicted_author_ids]

        # Style Accuracy berechnen
        correct_styles = sum(1 for p, t in zip(
            predicted_authors, true_authors) if p == t)
        style_accuracy = correct_styles / len(true_authors)

        # Perplexity berechnen
        perplexities = []
        for abstract in tqdm(generated_abstracts, desc="Berechne Perplexity"):
            ppl = predict_perplexity(abstract, ppl_model, ppl_tokenizer)
            perplexities.append(ppl)

        avg_ppl = sum(perplexities) / len(perplexities)

        duration = time.time() - start_time
        print(
            f"Ergebnis Alpha {alpha}: Style Acc = {style_accuracy:.2%}, Avg PPL = {avg_ppl:.2f} (Dauer: {duration/60:.1f} min)")

        tuning_results.append({
            "lambda_alpha": alpha,
            "style_accuracy": style_accuracy,
            "avg_perplexity": avg_ppl,
            "median_perplexity": sorted(perplexities)[len(perplexities)//2],
            "cross_entropy": statistics.mean(cross_entropy_loss),
            "runtime_seconds": duration
        })

    # 6. Ergebnisse speichern
    print("\nTuning beendet. Speichere Ergebnisse...")
    os.makedirs(f"{DATA_DIR}/steering_vectors/hyperparameter_tuning", exist_ok=True)
    df_results = pd.DataFrame(tuning_results)
    df_results.to_csv(
        f"{DATA_DIR}/steering_vectors/hyperparameter_tuning/lambda_opt_results.csv", index=False)
    print("Ergebnisse gespeichert unter: data/steering_vectors/hyperparameter_tuning/lambda_opt_results.csv")
    print(df_results)


def visualize_results(data_path: str):
    df = pd.read_csv(data_path)

    chosen_lambda = 0.875

    # 2. Den Plot aufbauen (2 untereinanderliegende Graphen)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Allgemeine Einstellungen für die vertikale Markierungslinie
    vline_kwargs = {'x': chosen_lambda, 'color': 'red', 'linestyle': '--',
                    'alpha': 0.7, 'label': f'Chosen: $\\lambda={chosen_lambda}$'}

    # --- Subplot 1: Style Accuracy (Higher is better) ---
    ax1.plot(df['lambda_alpha'], df['style_accuracy'],
             marker='o', color='royalblue', linewidth=2)
    ax1.axvline(**vline_kwargs)
    ax1.set_ylabel('Style Accuracy', fontsize=22)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=18)

    # --- Subplot 2: Cross Entropy & Perplexity (Lower is usually better) ---
    ax2.plot(df['lambda_alpha'], df['cross_entropy'], marker='s',
             color='darkorange', linewidth=2, label='Cross Entropy')
    ax2.plot(df['lambda_alpha'], df['avg_perplexity'], marker='^',
             color='forestgreen', linewidth=2, label='Avg Perplexity')
    ax2.axvline(**vline_kwargs)
    ax2.set_ylabel('Loss / Perplexity', fontsize=22)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=18)

    # --- Subplot 3: Runtime (Lower is better) ---
    # ax3.plot(df['lambda_alpha'], df['runtime_seconds'], marker='D', color='purple', linewidth=2)
    # ax3.axvline(**vline_kwargs)
    # ax3.set_ylabel('Runtime in Sekunden\n(Niedriger ist besser)', fontsize=12)
    # ax3.set_xlabel('$\\lambda$ (Lambda Alpha)', fontsize=12)
    # ax3.grid(True, linestyle=':', alpha=0.6)
    # ax3.legend()
    # ax3.set_title('Laufzeitverhalten', loc='left')

    # Layout optimieren und speichern/anzeigen
    plt.tight_layout()

    # Entkommentieren zum Speichern
    plt.savefig(
        f'{DATA_DIR}/steering_vectors/hyperparameter_tuning/lambda_opt_results.png', dpi=300)
    # plt.show()


if __name__ == "__main__":
    alphas_to_test = np.linspace(0.0, 2.0, num=17).tolist()

    run_tuning_pipeline(
        steering_vectors_path=f"{DATA_DIR}/steering_vectors/val_contrastive_own_18_19_20.safetensors",
        original_dataset_name="paper_dataset_with_summaries_unsloth_qwen3-4b-instruct-2507-unsloth-bnb-4bit",
        target_layer_indices=[18, 19, 20],
        alphas_to_test=alphas_to_test
    )

    visualize_results(
        f"{DATA_DIR}/steering_vectors/hyperparameter_tuning/lambda_opt_results.csv")
