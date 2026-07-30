from config import DATA_DIR
import gc
import json
import os
import time
from typing import cast

import torch
from tqdm import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from src.utils.papers import clean_latex, download_paper_src, get_paper_full_tex
from src.utils.llm import create_chunks, generate_text, parse_harmony_output, recursive_reduce, get_qwen_model
from src.utils.prompts import CHUNK_SYSTEM_PROMPT, CHUNK_USER_PROMPT, SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT


def generate_summary(model, tokenizer, paper_text: str):
    # 58k Tokens is too much, 3565 geht 6k gehen auf 3 GPUs verteilt
    chunks = create_chunks(paper_text, tokenizer)

    paper_time = time.time()

    if len(chunks) > 1:
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Verarbeite Chunk {i+1}/{len(chunks)}...")
            prompt = CHUNK_USER_PROMPT.format(text_chunk=chunk)

            summary = parse_harmony_output(generate_text(
                model, tokenizer, CHUNK_SYSTEM_PROMPT, prompt, 1024))
            summaries.append(summary)

            torch.cuda.empty_cache()
            gc.collect()

        # Wenn noch mehr als 5 chunks dann nochmal zusammenfassen (wollen ungefähr 5k tokens, 6 x 1024 > 5k Tokens)
        final_summaries = recursive_reduce(summaries, model, tokenizer)

        information = "\n\n".join(
            [f"Section {i+1} Summary:\n{s}" for i, s in enumerate(final_summaries)])
        task = "The following are summaries of a scientific paper's individual sections:"
    else:
        print("No Chunking needed. Proceeding to summarize directly from paper.")
        information = chunks[0]
        task = "The following is the paper you are required to summarize:\n\nPAPER START:"

    final_prompt = SUMMARY_USER_PROMPT.format(summaries=information, task=task)

    final_abstract = parse_harmony_output(generate_text(
        model, tokenizer, SUMMARY_SYSTEM_PROMPT, final_prompt, 4096))

    print(f"Zeit um die summary zu generieren: {time.time() - paper_time:.2f}s")

    return final_abstract


def create_paper_summaries(model, tokenizer, dataset_name="paper_dataset"):
    print("Starte erstellen von Summaries für ein Datenset.")

    loaded_data = load_from_disk(f"{DATA_DIR}/datasets/{dataset_name}")
    is_dataset_dict = isinstance(loaded_data, DatasetDict)
    if is_dataset_dict:
        dataset = concatenate_datasets(list(loaded_data.values()))
    else:
        dataset = loaded_data

    model_name = model.config.name_or_path.replace('/', '_')
    output_path = f"{DATA_DIR}/datasets/.temp/intermediate_results_summary_{dataset_name}_{model_name}.jsonl"
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_ids.add(entry['paper_id'])
                except json.JSONDecodeError:
                    pass  # Leere Zeilen oder Fehler ignorieren
        print(f"Bereits verarbeitete Papers gefunden: {len(processed_ids)}")

    # Stelle sicher, dass das Verzeichnis existiert
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'a', encoding='utf-8') as f_out:
        for row in tqdm(dataset):
            paper_id = row['paper_id']  # type: ignore
            author = row['author']  # type: ignore
            abstract = row['abstract']  # type: ignore

            print(f"\n--- Bearbeite Paper {paper_id} ---\n")
            if paper_id in processed_ids:
                print(f"Paper {paper_id} bereits vorhanden. Überspringe.")
                continue

            path = download_paper_src(paper_id, f"{DATA_DIR}/papers")
            if not path:
                continue

            paper_tex = get_paper_full_tex(path)
            if not paper_tex:
                continue

            paper_text = clean_latex(paper_tex)

            summary = generate_summary(model, tokenizer, paper_text)
            print(f"Summary generated for {paper_id}.")

            entry = {
                "paper_id": paper_id,
                "author": author,
                "summary": summary,
                "ground_truth": abstract,
            }

            f_out.write(json.dumps(entry) + "\n")
            f_out.flush()

    print("Generierung abgeschlossen. Rekonstruiere Splits...")

    full_processed_dataset = cast(Dataset, Dataset.from_json(output_path))

    if is_dataset_dict:
        final_dataset_dict = DatasetDict()
        loaded_dict = cast(DatasetDict, loaded_data)

        for split_name, split_dataset in loaded_dict.items():
            ids_in_split = set(split_dataset['paper_id'])

            final_dataset_dict[split_name] = full_processed_dataset.filter(
                lambda x: x['paper_id'] in ids_in_split
            )
            print(f"Split '{split_name}': {len(final_dataset_dict[split_name])} Einträge.")
        full_processed_dataset = final_dataset_dict

    print("Konvertiere in Hugging Face Dataset...")
    save_path = f"{DATA_DIR}/datasets/{dataset_name}_with_summaries_{model_name}"
    full_processed_dataset.save_to_disk(save_path)

    print(f"Finales DatasetDict erfolgreich unter {save_path} gespeichert.")


if __name__ == "__main__":
    model, tokenizer = get_qwen_model(load_in_4bit=True)
    create_paper_summaries(model, tokenizer, "unseen_paper_dataset")
