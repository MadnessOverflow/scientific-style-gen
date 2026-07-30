from config import DATA_DIR, MODELS_DIR
from datetime import datetime
from itertools import chain
import pandas as pd
from datasets import Dataset, DatasetDict

from src.datasets.latex_filter_papers import create_abstract_filter_list


PAPER_THRESHOLD = 5
DIFFERENT_TOPICS_THRESHOLD = 5
TOPICS = ['cs.AI', 'cs.LG']


def filter_authors(input_df: pd.DataFrame, topics=[], latex_filter=False, verbose=False):
    df = input_df.copy()
    df['categories'] = df['categories'].map(lambda c: c.strip().split(' '))
    df.rename(columns={'first_author': 'author',
              'id': 'paper_id'}, inplace=True)
    df['author'] = df['author'].str.strip()
    df['abstract'] = df['abstract'].str.strip()

    # Filter FIRST for authors with less than 6 different topics covered (simple method to try and exclude duplicate names)
    df_exploded = df.explode('categories')
    author_category_counts = df_exploded.groupby(
        'author')['categories'].nunique()
    authors_to_keep = author_category_counts[author_category_counts <=
                                             DIFFERENT_TOPICS_THRESHOLD].index.tolist()
    df = df[df['author'].isin(authors_to_keep)]

    # Filter for topics
    if len(topics) > 0:
        df = df[df['categories'].map(
            lambda c: any(topic in c for topic in topics))]

    # LaTeX filtering
    if latex_filter:
        try:
            filter_list_df = pd.read_csv(
                f'{DATA_DIR}/datasets/.temp/paper_filter_list.csv', dtype={'paper_id': str})
            ids_to_exclude = filter_list_df['paper_id'].unique()
            df = df[~df['paper_id'].isin(ids_to_exclude)]
        except FileNotFoundError:
            if verbose:
                print(
                    "Hinweis: 'paper_filter_list.csv' wurde nicht gefunden. Überspringe diesen Filter.")

    # Filter for min. number of papers per author
    author_groups = df.groupby('author')
    group_sizes = author_groups.size()
    df_filtered = author_groups.filter(lambda g: len(
        g) >= PAPER_THRESHOLD).reset_index(drop=True)

    if verbose:
        print(f"Anzahl der Authoren: {author_groups.ngroups}")
        print(
            f"Durchschnittliche Anzahl der Elemente pro Author: {group_sizes.mean()}")
        print(
            f"Median der Anzahl der Elemente pro Author: {group_sizes.median()}")

        print(
            f"\nAnzahl der Gruppen mit weniger als {PAPER_THRESHOLD} Elementen: {len(group_sizes[group_sizes < PAPER_THRESHOLD])}")
        print(
            f"Durchschnittliche Anzahl der Elemente pro Author nach mind. {PAPER_THRESHOLD} Paper Filter: {group_sizes[group_sizes >= PAPER_THRESHOLD].mean()}")
        print(
            f"Median der Anzahl der Elemente pro Author nach mind. {PAPER_THRESHOLD} Paper Filter: {group_sizes[group_sizes >= PAPER_THRESHOLD].median()}")

        print(
            f"\nInsgesamte Anzahl an übrigen Authoren: {len(group_sizes[group_sizes >= PAPER_THRESHOLD])}")
        print(
            f"Insgesamte Anzahl an übrigen Papern: {group_sizes[group_sizes >= PAPER_THRESHOLD].sum()}")

    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered[['paper_id', 'author', 'categories', 'abstract']]


def create_author_dataset(input_df: pd.DataFrame):
    def _get_abstract_length(abstract):
        return len(abstract.split(' '))

    df = input_df.copy()
    df['abstract_length'] = df['abstract'].map(_get_abstract_length)

    author_groups = df.groupby('author')
    author_dataset = author_groups.agg(
        num_of_papers=('paper_id', 'count'),
        avg_abstract_length=('abstract_length', lambda s: round(s.mean())),
        topics_covered=('categories', lambda s: list(
            set(chain.from_iterable(s)))),
        paper_list=('paper_id', list)
    )
    author_dataset.reset_index(inplace=True)
    author_dataset['author'] = author_dataset['author'].str.strip()
    author_dataset.insert(2, 'num_topics_covered',
                          author_dataset['topics_covered'].map(lambda t: len(t)))

    return author_dataset


def create_train_val_test_set(paper_df: pd.DataFrame):
    ds = Dataset.from_pandas(paper_df)
    # ds = ds.class_encode_column('author')

    train_test_ds = train_test_split(ds)

    train_val_ds = train_test_split(train_test_ds['train'])

    return DatasetDict({
        'train': train_val_ds['train'],
        'val': train_val_ds['test'],
        'test': train_test_ds['test']
    })


def train_test_split(ds: Dataset, sample_num=1):
    # Test set mit {sample_num} samples pro Author erstellen
    df_for_indexing = pd.DataFrame({'author': ds['author']})

    test_indices_df = df_for_indexing.groupby(
        'author').sample(n=sample_num, random_state=42)

    test_indices = test_indices_df.index.tolist()
    train_indices = df_for_indexing.drop(test_indices).index.tolist()

    train_ds = ds.select(train_indices)
    test_ds = ds.select(test_indices)

    return DatasetDict({
        'train': train_ds,
        'test': test_ds
    })


def save_dataset_statistics(
    main_df_filtered: pd.DataFrame,
    author_df: pd.DataFrame,
    dataset: DatasetDict,
    topics: list,
    output_filepath: str
):
    """
    Speichert wichtige Statistiken über das erstellte Dataset in einer Textdatei.
    """

    num_authors = len(author_df)
    num_papers_total = len(main_df_filtered)

    train_size = len(dataset['train'])
    val_size = len(dataset['val']) if 'val' in dataset else 0
    test_size = len(dataset['test'])

    avg_papers_per_author = author_df['num_of_papers'].mean()
    min_papers_per_author = author_df['num_of_papers'].min()
    max_papers_per_author = author_df['num_of_papers'].max()
    avg_abstract_length = author_df['avg_abstract_length'].mean()
    min_abstract_length = author_df['avg_abstract_length'].min()
    max_abstract_length = author_df['avg_abstract_length'].max()

    test_samples_per_author = 0
    if num_authors > 0:
        # Geht davon aus, dass die split-Logik (sample_num=1) konsistent ist
        test_samples_per_author = test_size // num_authors

    # Bericht als String erstellen
    report = []
    report.append("=========================================")
    report.append("     Dataset-Statistiken")
    report.append("=========================================")
    report.append(
        f"Bericht erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("--- Konfiguration ---")
    report.append(
        f"Verwendete Topics: {', '.join(topics) if topics else 'Alle'}")
    report.append(
        f"Min. Paper pro Author (PAPER_THRESHOLD): {PAPER_THRESHOLD}")
    report.append(
        f"Max. Topics pro Author (DIFFERENT_TOPICS_THRESHOLD): {DIFFERENT_TOPICS_THRESHOLD}\n")

    report.append("--- Gesamtübersicht ---")
    report.append(f"Anzahl einzigartiger Authoren (Klassen): {num_authors}")
    report.append(f"Anzahl gesamter Paper: {num_papers_total}\n")

    report.append("--- Author-Statistiken ---")
    report.append(
        f"Durchschnittliche Paper pro Author: {avg_papers_per_author:.2f}")
    report.append(
        f"Min/Max Paper pro Author: {min_papers_per_author} / {max_papers_per_author}")
    report.append(
        f"Durchschnittliche Abstract-Länge: {avg_abstract_length:.2f} Wörter")
    report.append(
        f"Min/Max Abstract-Länge: {min_abstract_length} / {max_abstract_length}\n")

    report.append("--- Train/Test Split ---")
    report.append(f"Größe Trainingsset: {train_size} Paper")
    report.append(f"Größe Validationset: {val_size} Paper")
    report.append(f"Größe Testset: {test_size} Paper")
    report.append(
        f"Verwendete Test-Samples pro Author: {test_samples_per_author}")
    report.append(
        f"Gesamt im DatasetDict: {train_size + val_size + test_size} Paper")

    report_str = "\n".join(report)

    # Bericht in Datei speichern
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"Statistiken erfolgreich in {output_filepath} gespeichert.")
    except IOError as e:
        print(f"Fehler beim Speichern der Statistiken: {e}")


if __name__ == '__main__':
    # Filter run 1
    main_df = pd.read_csv(f'{DATA_DIR}/datasets/.temp/arxiv_base_dataset.csv')

    filtered_df = filter_authors(main_df, topics=TOPICS)
    author_df = create_author_dataset(filtered_df)

    author_df.sort_values('num_of_papers', ascending=False, inplace=True)
    author_df.to_csv(f'{DATA_DIR}/datasets/.temp/author_dataset.csv', index=False)

    # Filter run 2 (filter using latex)
    # 2 filter runs because we want to reduce the number of downloads
    create_abstract_filter_list()

    filtered_df = filter_authors(main_df, topics=TOPICS, latex_filter=True)
    author_df = create_author_dataset(filtered_df)
    dataset = create_train_val_test_set(filtered_df)

    author_df.sort_values('num_of_papers', ascending=False, inplace=True)
    author_df.to_csv(f'{DATA_DIR}/datasets/.temp/author_dataset.csv', index=False)
    filtered_df.to_csv(f'{DATA_DIR}/datasets/.temp/paper_dataset.csv', index=False)

    dataset.save_to_disk(f'{DATA_DIR}/datasets/paper_dataset')

    save_dataset_statistics(
        main_df_filtered=filtered_df,
        author_df=author_df,
        dataset=dataset,
        topics=TOPICS,
        output_filepath=f'{DATA_DIR}/datasets/paper_dataset/dataset_statistics.txt'
    )
