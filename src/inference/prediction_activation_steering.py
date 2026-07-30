from config import DATA_DIR, MODELS_DIR
import math
import random
import time

import unsloth
import torch
import re
from typing import Callable, Dict, Any, cast
import pandas as pd
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm
from transformers import set_seed
from safetensors.torch import load_file

from src.utils.prompts import DEFAULT_ABSTRACT_SYSTEM_PROMPT, DEFAULT_ABSTRACT_USER_PROMPT
from src.utils.llm import get_qwen_model


def get_steering_hook(vector, alpha=1.0):
    """
    Erzeugt die Hook-Funktion, die den Steering Vector während der 
    Inferenz auf die Hidden States addiert.
    """
    def hook(module, inputs, output):
        hidden_states = output[0]

        # Vektor auf das richtige Gerät und den richtigen Datentyp bringen
        vec = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # Broadcasting: Vektor auf alle Tokens der Sequenz anwenden
        # Unterscheidung, ob der Vektor für Batches (dim=2) oder Single-Inference (dim=1) kommt
        if vec.dim() == 2:
            vec = vec.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        else:
            vec = vec.view(1, 1, -1)  # Fallback für [hidden_dim]

        # Steering-Vektor auf die Hidden States addieren
        modified_hidden_states = hidden_states + (alpha * vec)

        return (modified_hidden_states,) + output[1:]
    return hook


def _create_synthetic_abstracts_steered(
    model,
    tokenizer,
    format_inference_prompts: Callable[[Dict[str, Any]], Dict[str, str]],
    file_name: str,
    dataset_name: str,
    steering_vectors_path: str,
    alpha=1.0,
    target_layer_indices: list[int] = [-1]  # -1 = last layer
):
    """
    Erstellt synthetische Abstracts mittels sequentieller Generierung (Single-Inference)
    und wendet Activation Steering pro Paper an.
    """

    dataset_path = f"{DATA_DIR}/datasets/{dataset_name}"

    print(f"Lade Test-Daten aus: {dataset_path}")
    loaded_data = load_from_disk(dataset_path)
    if isinstance(loaded_data, DatasetDict):
        if "test" in loaded_data:
            dataset = loaded_data["test"]
        else:
            print(f"Hinweis: Split 'test' nicht gefunden.")
            return
    else:
        dataset = loaded_data

    print(f"Lade Steering Vektoren aus: {steering_vectors_path}")
    try:
        steering_vectors = load_file(steering_vectors_path)
    except Exception as e:
        print(f"Fehler beim Laden der Steering Vektoren: {e}")
        return

    # Create additional random samples
    all_summaries = dataset["summary"]
    all_authors = dataset["author"]
    all_paper_ids = dataset["paper_id"]

    def get_random_summaries(example):
        current_author = example["author"]
        paper_id = example["paper_id"]

        random.seed(paper_id)

        # Kandidaten finden (alle Summaries, außer die des aktuellen Autors)
        candidates = [
            (s, pid) for s, a, pid in zip(all_summaries, all_authors, all_paper_ids)
            if a != current_author
        ]

        random_samples = random.sample(candidates, 2)

        return {
            "random_summary_1": random_samples[0][0],
            "random_summary_2": random_samples[1][0],
            "random_summary_1_id": random_samples[0][1],
            "random_summary_2_id": random_samples[1][1],
        }
    print("Füge random summaries hinzu...")
    dataset = dataset.map(get_random_summaries)

    print("Formatiere Prompts...")
    dataset = dataset.map(format_inference_prompts)

    print(
        f"Starte sequentielle Generierung (Steering Alpha: {alpha}) für {len(dataset)} Einträge...")

    set_seed(67)  # Optional, falls gewünscht

    prompt_example = None
    answer_example = None

    generated_main = []
    generated_rnd1 = []
    generated_rnd2 = []

    prompt_types = {
        "text_prompt": generated_main,
        "text_prompt_random_1": generated_rnd1,
        "text_prompt_random_2": generated_rnd2
    }

    for row in tqdm(dataset, desc="Inference"):
        paper_id = row["paper_id"]  # type: ignore

        paper_layer_vectors = {}

        for layer_idx in target_layer_indices:
            dict_key = f"{paper_id}_layer_{layer_idx}"
            if dict_key in steering_vectors:
                paper_layer_vectors[layer_idx] = steering_vectors[dict_key]
            else:
                break

        for p_key, target_list in prompt_types.items():
            prompt = row[p_key]  # type: ignore

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=4096
            ).to(model.device)

            input_ids = inputs["input_ids"]

            if not prompt_example:
                prompt_example = tokenizer.decode(
                    input_ids[0], skip_special_tokens=False)

            active_hooks = []
            for layer_idx in target_layer_indices:
                target_layer = model.model.layers[layer_idx]
                vector_for_layer = paper_layer_vectors[layer_idx]

                hook_handle = target_layer.register_forward_hook(
                    get_steering_hook(vector_for_layer, alpha=alpha))
                active_hooks.append(hook_handle)

            try:
                with torch.inference_mode():
                    outputs = model.generate(
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
                        eos_token_id=tokenizer.eos_token_id,
                    )
            finally:
                for hook in active_hooks:
                    hook.remove()

            input_length = input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]

            decoded_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True)

            if not answer_example:
                answer_example = decoded_text

            # Cleanup
            cleaned_text = re.sub(r"\*\*abstract\*\*\s*",
                                  "", decoded_text, flags=re.IGNORECASE).strip()
            target_list.append(cleaned_text)

    print("Generierung beendet")
    print("-" * 30)
    print("Sanity Check:")
    print(f"Prompt: {prompt_example}...")
    print(f"Answer: {answer_example}...")

    dataset = dataset.add_column(
        name="gen_abstract", column=generated_main, new_fingerprint=None)  # type: ignore
    dataset = dataset.add_column(name="gen_abstract_random_1",
                                 column=generated_rnd1, new_fingerprint=None)  # type: ignore
    dataset = dataset.add_column(name="gen_abstract_random_2",
                                 column=generated_rnd2, new_fingerprint=None)  # type: ignore

    cols_to_remove = [
        k for k in prompt_types.keys() if k in dataset.column_names]
    dataset = dataset.remove_columns(cols_to_remove)

    output_dir = f"{DATA_DIR}/abstract_predictions/{file_name}_steered"
    print(f"Speichere finales Dataset nach: {output_dir}")
    dataset.save_to_disk(output_dir)

    df = cast(pd.DataFrame, dataset.to_pandas())
    df[['gen_abstract', 'gen_abstract_random_1', 'gen_abstract_random_2']].to_csv(
        f'{DATA_DIR}/datasets/.temp/{file_name}_steered.csv', index=False)
    print("Fertig.")


def create_synthetic_abstracts_steered(
    model,
    tokenizer,
    model_name: str,
    dataset_name: str,
    steering_vectors_path: str,
    alpha: float,
    target_layer_indices: list[int]
):
    def format_inference_prompts(example):
        keys = ["summary", "random_summary_1", "random_summary_2"]
        results = {}

        for key in keys:
            messages = [
                {"role": "system", "content": DEFAULT_ABSTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": DEFAULT_ABSTRACT_USER_PROMPT.format(
                    summary=example[key])},
            ]

            output_key = "text_prompt" if key == "summary" else f"text_prompt_{key.replace('summary_', '')}"
            results[output_key] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        return results

    _create_synthetic_abstracts_steered(
        model=model,
        tokenizer=tokenizer,
        format_inference_prompts=format_inference_prompts,
        file_name=f"{model_name}_abstracts_dataset",
        dataset_name=dataset_name,
        steering_vectors_path=steering_vectors_path,
        alpha=alpha,
        target_layer_indices=target_layer_indices
    )


if __name__ == "__main__":
    TARGET_LAYERS = [18, 19, 20]
    APPROACH = "own"  # "own" oder "paper"

    model, tokenizer = get_qwen_model()
    start_time = time.time()
    create_synthetic_abstracts_steered(
        model, tokenizer, f"Unseen_Qwen3_{APPROACH}_{('_').join(map(str, TARGET_LAYERS))}",
        "unseen_paper_dataset_with_summaries_unsloth_qwen3-4b-instruct-2507-unsloth-bnb-4bit",
        f"{DATA_DIR}/steering_vectors/unseen_test_contrastive_{APPROACH}_{('_').join(map(str, TARGET_LAYERS))}.safetensors",
        alpha=0.875,
        target_layer_indices=TARGET_LAYERS
    )
    duration = time.time() - start_time
    print(
        f"Prediction hat {math.floor(duration / 60)}min und {duration % 60}s gebraucht.")
