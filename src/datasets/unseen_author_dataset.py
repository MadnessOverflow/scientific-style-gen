from config import DATA_DIR, MODELS_DIR
import pandas as pd
from datasets import Dataset

from src.datasets.latex_filter_papers import create_abstract_filter_list
from src.datasets.author_dataset import create_author_dataset, save_dataset_statistics, train_test_split

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
                f'{DATA_DIR}/datasets/.temp/paper_filter_list_unseen_authors.csv', dtype={'paper_id': str})
            ids_to_exclude = filter_list_df['paper_id'].unique()
            df = df[~df['paper_id'].isin(ids_to_exclude)]
        except FileNotFoundError:
            if verbose:
                print(
                    "Hinweis: 'paper_filter_list.csv' wurde nicht gefunden. Überspringe diesen Filter.")

    # Filter for min. number of papers per author
    author_groups = df.groupby('author')
    df_filtered = author_groups.filter(lambda g: (
        len(g) < 5 and len(g) >= 4)).reset_index(drop=True)

    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered[['paper_id', 'author', 'categories', 'abstract']]


if __name__ == '__main__':
    # Filter run 1
    main_df = pd.read_csv(f'{DATA_DIR}/datasets/.temp/arxiv_base_dataset.csv')

    filtered_df = filter_authors(main_df, topics=TOPICS)
    author_df = create_author_dataset(filtered_df)

    author_df.sort_values('num_of_papers', ascending=False, inplace=True)
    # author_df.to_csv(f'{DATA_DIR}/datasets/.temp/unseen_author_dataset.csv', index=False)

    # Filter run 2 (filter using latex)
    # 2 filter runs because we want to reduce the number of downloads
    create_abstract_filter_list(f'{DATA_DIR}/datasets/.temp/unseen_author_dataset.csv',
                                'paper_filter_list_unseen_authors'
    )

    filtered_df = filter_authors(main_df, topics=TOPICS, latex_filter=True)
    author_df = create_author_dataset(filtered_df)

    author_df.sort_values('num_of_papers', ascending=False, inplace=True)
    filtered_df.to_csv(f'{DATA_DIR}/datasets/.temp/unseen_paper_dataset.csv', index=False)
    author_df.to_csv(f'{DATA_DIR}/datasets/.temp/unseen_author_dataset.csv', index=False)

    dataset = train_test_split(Dataset.from_pandas(filtered_df))
    dataset.save_to_disk(f'{DATA_DIR}/datasets/unseen_paper_dataset')

    save_dataset_statistics(
        main_df_filtered=filtered_df,
        author_df=author_df,
        dataset=dataset,
        topics=TOPICS,
        output_filepath=f'{DATA_DIR}/datasets/unseen_paper_dataset/dataset_statistics.txt'
    )
