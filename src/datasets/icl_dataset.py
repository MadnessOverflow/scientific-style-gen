from config import DATA_DIR, MODELS_DIR
import random
from typing import cast

import pandas as pd
from tqdm import tqdm
from datasets import DatasetDict, Dataset, concatenate_datasets, load_from_disk


def create_icl_dataset(base_dataset_name: str, author_dataset_name: str, output_name: str):
    loaded_data = load_from_disk(f"{DATA_DIR}/datasets/{base_dataset_name}")
    author_df = pd.read_csv(f"{DATA_DIR}/datasets/.temp/{author_dataset_name}.csv")
    paper_ids_lookup = dict(zip(author_df['author'], author_df['paper_list']))

    if isinstance(loaded_data, DatasetDict):
        dataset = concatenate_datasets(list(loaded_data.values()))
    else:
        dataset = loaded_data

    abstract_lookup = {item['paper_id']: item['ground_truth']
                       for item in dataset}  # type: ignore

    dataset_rows = []

    for row in tqdm(dataset):
        paper_id = row['paper_id']  # type: ignore
        author = row['author']  # type: ignore
        gt_abstract = row['ground_truth']  # type: ignore
        summary = row['summary']  # type: ignore

        print(f"\n--- Bearbeite Paper {paper_id} ---\n")

        candidates = [p for p in eval(
            paper_ids_lookup[author]) if p != paper_id]
        random.seed(paper_id)

        if len(candidates) >= 3:
            selection_ids = random.sample(candidates, 3)
        else:
            print(
                f"Warning: Author '{author}' has not enough papers. (num of papers: {len(candidates)})")
            return

        selection_abstracts = []
        for p_id in selection_ids:
            abstract = abstract_lookup.get(p_id, None)
            if abstract:
                selection_abstracts.append(abstract)

        if not selection_abstracts:
            continue

        entry = {
            "paper_id": paper_id,
            "author": author,
            "summary": summary,
            "ground_truth": gt_abstract,
            "example_1": selection_abstracts[0].strip() if len(selection_abstracts) > 0 else "",
            "example_2": selection_abstracts[1].strip() if len(selection_abstracts) > 1 else "",
            "example_3": selection_abstracts[2].strip() if len(selection_abstracts) > 2 else "",
            "example_1_id": selection_ids[0],
            "example_2_id": selection_ids[1],
            "example_3_id": selection_ids[2]
        }

        dataset_rows.append(entry)

    print("Generierung abgeschlossen. Rekonstruiere Splits...")

    full_processed_dataset = cast(
        Dataset, Dataset.from_pandas(pd.DataFrame(dataset_rows)))

    if isinstance(loaded_data, DatasetDict):
        final_dataset_dict = DatasetDict()

        for split_name in loaded_data:
            ids_in_split = set(loaded_data[split_name]['paper_id'])

            final_dataset_dict[split_name] = full_processed_dataset.filter(
                lambda x: x['paper_id'] in ids_in_split
            )
            print(
                f"Split '{split_name}': {len(final_dataset_dict[split_name])} Einträge.")
        full_processed_dataset = final_dataset_dict

    print("Konvertiere in Hugging Face Dataset...")
    save_path = f"{DATA_DIR}/datasets/{output_name}"
    full_processed_dataset.save_to_disk(save_path)

    print(f"Finales DatasetDict erfolgreich unter {save_path} gespeichert.")


if __name__ == "__main__":
    create_icl_dataset("unseen_paper_dataset_with_summaries_unsloth_qwen3-4b-instruct-2507-unsloth-bnb-4bit",
                       "unseen_author_dataset", "unseen_paper_dataset_icl")
