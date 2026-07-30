from config import DATA_DIR, MODELS_DIR
import re
from typing import Any, Callable, Dict
import torch
from tqdm import tqdm

from src.utils.llm import get_qwen_model
from src.utils.prompts import DEFAULT_ABSTRACT_SYSTEM_PROMPT, DEFAULT_ABSTRACT_USER_PROMPT

from datasets import DatasetDict, load_from_disk
from transformers import set_seed


def _create_synthetic_abstracts(model, tokenizer, format_inference_prompts: Callable[[Dict[str, Any]], Dict[str, str]], file_name: str, dataset_name: str):
    """
    Erstellt synthetische Abstracts für das gesamte Datenset mittels sequentieller Generierung (Single-Inference).
    Verzichtet komplett auf Padding und Batch-Inferenz, um Artefakte zu vermeiden.
    """

    dataset_path = f"{DATA_DIR}/datasets/{dataset_name}"

    print(f"Lade Daten aus: {dataset_path}")
    loaded_data = load_from_disk(dataset_path)

    is_dataset_dict = isinstance(loaded_data, DatasetDict)

    if not is_dataset_dict:
        loaded_data = DatasetDict({"all": loaded_data})

    print("Formatiere Prompts...")
    loaded_data = loaded_data.map(format_inference_prompts)

    prompt_example = None
    answer_example = None

    for split_name, dataset in loaded_data.items():
        print(
            f"Starte sequentielle Generierung für Split '{split_name}' mit {len(dataset)} Einträgen...")
        generated_main = []

        prompt_types = {
            "text_prompt": generated_main,
        }

        set_seed(67)  # Optional, falls gewünscht

        for row in tqdm(dataset, desc=f"Inference {split_name}"):
            for p_key, target_list in prompt_types.items():
                prompt = row[p_key]  # type: ignore

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=8192
                ).to("cuda")

                input_ids = inputs["input_ids"]

                if not prompt_example:
                    prompt_example = tokenizer.decode(
                        input_ids[0], skip_special_tokens=False)

                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=1024,  # Längstes Abstract im Test-Split: 511 Tokens
                        do_sample=True,
                        use_cache=True,
                        temperature=0.7,
                        top_p=0.8,
                        top_k=20,
                        min_p=0,
                        repetition_penalty=1.05,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                input_length = input_ids.shape[1]
                generated_tokens = outputs[0][input_length:]

                decoded_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True)

                if not answer_example:
                    answer_example = decoded_text

                # Cleanup
                cleaned_text = re.sub(
                    r"\*\*abstract\*\*\s*", "", decoded_text, flags=re.IGNORECASE).strip()
                target_list.append(cleaned_text)

        loaded_data[split_name] = dataset.add_column(
            name="gen_abstract", column=generated_main, new_fingerprint=None)  # type: ignore
        cols_to_remove = ["text_prompt"]
        loaded_data[split_name] = loaded_data[split_name].remove_columns(
            cols_to_remove)

    print("Generierung beendet")
    print("-" * 30)
    print("Sanity Check:")
    print(f"Prompt: {prompt_example}...")
    print(f"Answer: {answer_example}...")

    if not is_dataset_dict:
        final_dataset = loaded_data["all"]
    else:
        final_dataset = loaded_data

    output_dir = f"{DATA_DIR}/datasets/{file_name}"
    print(f"Speichere finales Dataset nach: {output_dir}")
    final_dataset.save_to_disk(output_dir)

    print("Fertig.")


def create_synthetic_abstracts_full(model, tokenizer, model_name: str, dataset_name: str):
    def format_inference_prompts(example):
        keys = ["summary"]
        results = {}

        for key in keys:
            messages = [
                {"role": "system", "content": DEFAULT_ABSTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": DEFAULT_ABSTRACT_USER_PROMPT.format(
                    summary=example[key])},
            ]

            output_key = "text_prompt"
            results[output_key] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        return results

    _create_synthetic_abstracts(model, tokenizer, format_inference_prompts,
                                f"{model_name}_abstracts_dataset_full", dataset_name)


if __name__ == "__main__":
    model, tokenizer = get_qwen_model()
    create_synthetic_abstracts_full(model, tokenizer, "Qwen3_unseen",
                                    "unseen_paper_dataset_with_summaries_unsloth_qwen3-4b-instruct-2507-unsloth-bnb-4bit")
