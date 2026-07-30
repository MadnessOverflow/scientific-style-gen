from config import DATA_DIR, MODELS_DIR
import pandas as pd

from src.evaluation.compare_content import get_ids_from_dataset
from src.utils.papers import download_paper_src, get_introduction, get_paper_full_tex


def create_abstract_filter_list(author_dataset_path=f"{DATA_DIR}/datasets/.temp/author_dataset.csv", filter_list_name="paper_filter_list"):
    print("Starte Validierung der Abstracts im Datenset auf Korrektheit.")
    ids_dict = get_ids_from_dataset(author_dataset_path)
    print(
        f"Alle IDs aus dem Datenset geladen: {list(ids_dict.items())[:3]}...")

    filter_list = []
    num_of_papers = 0

    for current_author, paper_ids in ids_dict.items():
        print(f"\n------\nPrüfe Abstracts für Autor: {current_author}\n")
        num_of_papers += len(paper_ids)

        for paper_id in paper_ids:
            print(f"--- Analysiere Paper {paper_id} ---")

            path = download_paper_src(paper_id, f"{DATA_DIR}/papers")
            if not path:
                print(f" -> Paper nicht gefunden: {paper_id}")
                filter_list.append({
                    "paper_id": paper_id,
                    "reason": "paper_not_found"
                })
                continue

            paper_tex = get_paper_full_tex(path)
            if not paper_tex:
                print(f" -> Kein TeX-Source gefunden: {paper_id}")
                filter_list.append({
                    "paper_id": paper_id,
                    "reason": "paper_tex_not_found"
                })
                continue

            introduction = get_introduction(paper_tex)
            if not introduction:
                print(f" -> Keine Introduction gefunden: {paper_id}")
                filter_list.append({
                    "paper_id": paper_id,
                    "reason": "introduction_not_found"
                })
                continue

    print(f"Num of papers before filtering: {num_of_papers}.")
    print(
        f"\nSpeichere {len(filter_list)} gefilterte Einträge in '{filter_list_name}.csv'...")
    df = pd.DataFrame(filter_list)
    df.to_csv(f'{DATA_DIR}/datasets/.temp/{filter_list_name}.csv', index=False)
    print("Fertig.")


if __name__ == "__main__":
    create_abstract_filter_list()
