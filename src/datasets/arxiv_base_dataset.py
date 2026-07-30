from config import DATA_DIR, MODELS_DIR
from datetime import datetime
from typing import cast
from datasets import load_dataset, Dataset
import pandas as pd


def filter_licences(input_df: pd.DataFrame):
    df = input_df.copy()
    return df[~(df['license'].isin(['http://arxiv.org/licenses/nonexclusive-distrib/1.0/', 'http://creativecommons.org/licenses/by-nc-nd/4.0/', None]))]


def filter_release_year(input_df: pd.DataFrame):
    def _check_version_year(versions_list):
        for item in versions_list:
            if item.get('version') == 'v1':
                created_date = datetime.strptime(
                    item.get('created'), '%a, %d %b %Y %H:%M:%S %Z')
                if created_date.year <= 2022:
                    return True
        return False

    df = input_df.copy()
    df = df[df['versions'].apply(_check_version_year)]
    return df


def add_first_author_col(input_df: pd.DataFrame):
    def _get_first_author(author_list):
        if len(author_list) == 0:
            return ""
        first_author = author_list[0]

        return f"{first_author[0]} {first_author[1]} {first_author[2]}"

    df = input_df.copy()
    df['first_author'] = df['authors_parsed'].map(_get_first_author)
    df = df[~df['first_author'].str.contains(
        'collaboration', case=False, na=False)]
    return df


if __name__ == '__main__':
    dataset = load_dataset("librarian-bots/arxiv-metadata-snapshot")

    df = cast(pd.DataFrame, cast(Dataset, dataset['train']).to_pandas())
    df = filter_licences(df)
    df = filter_release_year(df)
    df = add_first_author_col(df)
    print(df.head())
    df.to_csv(f'{DATA_DIR}/datasets/.temp/arxiv_base_dataset.csv', index=False)
