from config import DATA_DIR, MODELS_DIR
import gc
import json
import os
import random
import re
import time
from typing import Any, Callable, Dict, List, cast
import pandas as pd
import torch
from tqdm import tqdm

from src.training.lora_baseline import get_lora_model
from src.utils.papers import clean_latex, download_paper_src, get_paper_full_tex
from src.utils.llm import generate_text, parse_harmony_output
from src.utils.prompts import ICL_SYSTEM_PROMPT, ICL_USER_PROMPT, LORA_SYSTEM_PROMPT, LORA_USER_PROMPT, DEFAULT_ABSTRACT_SYSTEM_PROMPT, DEFAULT_ABSTRACT_USER_PROMPT

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from transformers import set_seed


def generate_abstract_icl(model, tokenizer, syn_abstract: str, examples: List[str]):
    combined_examples = "\n\n".join(
        [f"Example {i+1}: \n{s}" for i, s in enumerate(examples)])

    final_prompt = ICL_USER_PROMPT.format(
        examples=combined_examples, synthetic_abstract=syn_abstract)

    print(f"Finaler prompt: {final_prompt}")

    final_abstract = parse_harmony_output(generate_text(
        model, tokenizer, ICL_SYSTEM_PROMPT, final_prompt, 4096))

    return final_abstract



def _create_synthetic_abstracts(model, tokenizer, format_inference_prompts: Callable[[Dict[str, Any]], Dict[str, str]], file_name: str, dataset_name="paper_dataset_with_summaries_unsloth_Qwen3-4B-Instruct-2507"):
    """
    Erstellt synthetische Abstracts mittels sequentieller Generierung (Single-Inference).
    Verzichtet komplett auf Padding und Batch-Inferenz, um Artefakte zu vermeiden.
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

    print(f"Starte sequentielle Generierung für {len(dataset)} Einträge...")
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

    set_seed(67)  # Optional, falls gewünscht

    for row in tqdm(dataset, desc="Inference"):
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

    output_dir = f"{DATA_DIR}/abstract_predictions/{file_name}"
    print(f"Speichere finales Dataset nach: {output_dir}")
    dataset.save_to_disk(output_dir)

    df = cast(pd.DataFrame, dataset.to_pandas())
    df[['gen_abstract', 'gen_abstract_random_1', 'gen_abstract_random_2']].to_csv(
        f'{DATA_DIR}/datasets/.temp/{file_name}.csv', index=False)
    print("Fertig.")


def create_synthetic_abstracts(model, tokenizer, model_name: str, dataset_name: str):
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

    _create_synthetic_abstracts(
        model, tokenizer, format_inference_prompts, f"{model_name}_abstracts_dataset", dataset_name)


def create_synthetic_abstracts_icl(model, tokenizer, model_name: str, dataset_name: str):
    def format_inference_prompts(example):
        abstract_examples = [example['example_1'],
                             example['example_2'], example['example_3']]

        keys = ["summary", "random_summary_1", "random_summary_2"]
        results = {}

        for key in keys:
            messages = [
                {"role": "system", "content": ICL_SYSTEM_PROMPT},
                {"role": "user", "content":
                    ICL_USER_PROMPT.format(
                        examples="\n\n".join(
                            [f"Example {i+1}: \n{s}" for i, s in enumerate(abstract_examples)]),
                        summary=example[key]
                    )
                 },
            ]

            output_key = "text_prompt" if key == "summary" else f"text_prompt_{key.replace('summary_', '')}"
            results[output_key] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        return results

    _create_synthetic_abstracts(model, tokenizer,
                                format_inference_prompts,
                                f"{model_name}_icl_abstracts_dataset",
                                dataset_name)


def create_synthetic_abstracts_lora(model, tokenizer, model_name: str, dataset_name: str):
    def format_inference_prompts(example):
        keys = ["summary", "random_summary_1", "random_summary_2"]
        results = {}

        for key in keys:
            messages = [
                {"role": "system", "content": LORA_SYSTEM_PROMPT},
                {"role": "user", "content":
                    LORA_USER_PROMPT.format(
                        author=example["author"],
                        summary=example[key]
                    ),
                 }
            ]

            output_key = "text_prompt" if key == "summary" else f"text_prompt_{key.replace('summary_', '')}"
            results[output_key] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        return results

    _create_synthetic_abstracts(
        model, tokenizer, format_inference_prompts, f"{model_name}_abstracts_dataset", dataset_name)


if __name__ == "__main__":
    # model, tokenizer = get_lora_model("Qwen_Qwen3-4B-Instruct-2507")
    # create_synthetic_abstracts_lora(model, tokenizer, "Qwen3_LoRA", "paper_dataset_with_summaries_unsloth_Qwen3-4B-Instruct-2507")

    # model, tokenizer = get_qwen_model()
    model, tokenizer = get_lora_model("Qwen3-4B-Instruct-2507_icl")
    create_synthetic_abstracts_icl(
        model, tokenizer, "Unseen_Qwen3_LoRA", "unseen_paper_dataset_icl")

    # model, tokenizer = get_qwen_model()
    # create_synthetic_abstracts(model, tokenizer, "Unseen_Qwen3", "unseen_paper_dataset_with_summaries_unsloth_qwen3-4b-instruct-2507-unsloth-bnb-4bit")

    # model, tokenizer = get_qwen_model(load_in_4bit=True)
    # create_paper_summaries(model, tokenizer, "unseen_paper_dataset")

    # --- Einzelnes Paper überprüfen: ---
    # paper_id = "2206.06629"

    # papers = pd.read_csv(f'{DATA_DIR}/filtered_dataset.csv').set_index("id")
    # print(f"\n--- Bearbeite Paper {paper_id} ---\n")

    # path = download_paper_src(paper_id, f"{DATA_DIR}/papers")
    # if not path: exit()

    # paper_tex = get_paper_full_tex(path)
    # if not paper_tex: exit()

    # abstract = papers.loc[paper_id, "abstract"]

    # print(f"\n\nAbstract: \n{abstract}\n")

    # paper_text = clean_latex(paper_tex)

    # model, tokenizer = get_gwen_model()
    # gen_abstract = generate_abstract(model, tokenizer, paper_text)
    # print(gen_abstract)

    # with open(f"{DATA_DIR}/tmp_paper.md", "w", encoding="utf-8") as file:
    #     file.write(paper_text)
