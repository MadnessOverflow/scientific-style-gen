from config import DATA_DIR, MODELS_DIR
import math
import random
import re
import time

import unsloth
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm
from transformers import set_seed

from src.style_control_network.implementation.end_to_end_model import EndToEndSteeredLLM
from src.utils.prompts import DEFAULT_ABSTRACT_SYSTEM_PROMPT, DEFAULT_ABSTRACT_USER_PROMPT


def create_synthetic_abstracts_style_control_network(
    model: EndToEndSteeredLLM,
    model_name: str,
    dataset_name: str,
):
    """
    Erstellt synthetische Abstracts mittels des trainierten Hypernetworks.

    Args:
        model: Initialisiertes EndToEndSteeredLLM (mit geladenem Hypernet).
        model_name: Name für den Output-Dateinamen.
        dataset_name: Name des Eingabe-Datasets unter data/datasets/.
    """
    file_name = f"{model_name}_hyper_steering_abstracts_dataset"
    dataset_path = f"{DATA_DIR}/datasets/{dataset_name}"

    print(f"Lade Test-Daten aus: {dataset_path}")
    loaded_data = load_from_disk(dataset_path)
    if isinstance(loaded_data, DatasetDict):
        if "test" in loaded_data:
            dataset = loaded_data["test"]
        else:
            print("Hinweis: Split 'test' nicht gefunden.")
            return
    else:
        dataset = loaded_data

    # Zufällige Summaries von anderen Autoren hinzufügen
    all_summaries = dataset["summary"]
    all_authors = dataset["author"]
    all_paper_ids = dataset["paper_id"]

    def get_random_summaries(example):
        current_author = example["author"]
        paper_id = example["paper_id"]

        random.seed(paper_id)

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

    print(f"Starte sequentielle Generierung für {len(dataset)} Einträge...")

    set_seed(67)

    prompt_example = None
    answer_example = None

    generated_main = []
    generated_rnd1 = []
    generated_rnd2 = []

    prompt_types = [
        ("summary",          generated_main),
        ("random_summary_1", generated_rnd1),
        ("random_summary_2", generated_rnd2),
    ]

    tokenizer = model.tokenizer

    for row in tqdm(dataset, desc="Hypernet Inference"):
        # Die 3 Stil-Abstracts des Autors für das Hypernetwork
        author_abstracts = [row["example_1"],
                            row["example_2"], row["example_3"]]  # type: ignore

        for summary_key, target_list in prompt_types:
            summary = row[summary_key]  # type: ignore

            # Prompt formatieren
            messages = [
                {"role": "system", "content": DEFAULT_ABSTRACT_SYSTEM_PROMPT},
                {"role": "user",   "content": DEFAULT_ABSTRACT_USER_PROMPT.format(
                    summary=summary)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if not prompt_example:
                prompt_example = prompt

            # Generierung über das Hypernetwork (setzt intern die Hooks und räumt sie wieder auf)
            decoded_text = model.generate_styled_abstract(
                prompt=prompt,
                author_abstracts=author_abstracts,
                max_new_tokens=1024,
            )

            if not answer_example:
                answer_example = decoded_text

            cleaned_text = re.sub(r"\*\*abstract\*\*\s*",
                                  "", decoded_text, flags=re.IGNORECASE).strip()
            target_list.append(cleaned_text)

    print("Generierung beendet")
    print("-" * 30)
    print("Sanity Check:")
    print(f"Prompt: {prompt_example}...")
    print(f"Answer: {answer_example}...")

    # Spalten hinzufügen
    dataset = dataset.add_column(
        name="gen_abstract",          column=generated_main, new_fingerprint=None)  # type: ignore
    dataset = dataset.add_column(name="gen_abstract_random_1",
                                 column=generated_rnd1, new_fingerprint=None)  # type: ignore
    dataset = dataset.add_column(name="gen_abstract_random_2",
                                 column=generated_rnd2, new_fingerprint=None)  # type: ignore

    output_dir = f"{DATA_DIR}/abstract_predictions/{file_name}"
    print(f"Speichere finales Dataset nach: {output_dir}")
    dataset.save_to_disk(output_dir)

    import pandas as pd
    from typing import cast
    df = cast(pd.DataFrame, dataset.to_pandas())
    df[["gen_abstract", "gen_abstract_random_1", "gen_abstract_random_2"]].to_csv(
        f"{DATA_DIR}/datasets/.temp/{file_name}.csv", index=False
    )
    print("Fertig.")


if __name__ == "__main__":
    start_time = time.time()

    model = EndToEndSteeredLLM(
        target_layers=[18, 19, 20],  # list(range(36)), #
        alpha=1.0,
        pretrained_style_control_network_path=f"{MODELS_DIR}/style_control_network/hyp_lora_3_lay/style_control_network.pt",
        steering_method="lora_l",
        style_control_network_kwargs={
            "embedder_name": "AIDA-UPM/star",
        }
    ).eval()

    create_synthetic_abstracts_style_control_network(
        model=model,
        model_name="Qwen3_lora_3_lay",
        dataset_name="paper_dataset_icl",
    )

    duration = time.time() - start_time
    print(
        f"Prediction hat {math.floor(duration / 60)}min und {duration % 60:.1f}s gebraucht.")
