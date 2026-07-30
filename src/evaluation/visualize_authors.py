from config import DATA_DIR, MODELS_DIR
import ast
from typing import Counter
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    df = pd.read_csv(f'{DATA_DIR}/author_dataset.csv')
    print(df.head())
    df.info()

    df['topics_covered'] = df['topics_covered'].apply(ast.literal_eval)

    # Histogramm zur Verteilung der Paper pro Author
    paper_counts = df['num_of_papers'].value_counts(
    ).sort_index(ascending=False)
    cumulative_authors = paper_counts.cumsum().sort_index(ascending=True)

    fig, ax1 = plt.subplots(figsize=(14, 8))
    sns.histplot(df['num_of_papers'], discrete=True,
                 ax=ax1, color='steelblue')  # type: ignore
    ax1.set_yscale('log')
    ax1.set_xlabel('Anzahl der Paper pro Autor', fontsize=12)
    ax1.set_ylabel('Anzahl der Autoren (log)', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.plot(cumulative_authors.index, cumulative_authors.values.tolist(),
             color='tomato', marker='.', linestyle='-')
    ax2.set_ylabel('Kumulative Anzahl an Autoren (>= X Paper)',
                   fontsize=12, color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')
    # ax2.fill_between(cumulative_authors.index, cumulative_authors.values.tolist(), alpha=0.15, color='tomato')

    plt.title('Verteilung und kumulative Summe der Paper pro Autor', fontsize=16)
    fig.tight_layout()
    plt.savefig('papers_per_author_distribution.png')
    plt.close()

    # Verteilung der durchschnittlichen Abstract Länge
    plt.figure(figsize=(12, 7))
    sns.histplot(df['avg_abstract_length'], bins=40,
                 kde=True, color='skyblue')  # type: ignore
    plt.title('Verteilung der durchschnittlichen Abstract-Länge', fontsize=16)

    plt.savefig('abstract_length_distribution.png')
    plt.close()

    # --- THEMEN --- #
    # Themen Bar Chart Allg.
    plt.figure(figsize=(12, 7))
    all_topics = [topic for sublist in df['topics_covered']
                  for topic in sublist]
    topic_counts = Counter(all_topics).most_common(20)
    topics_df = pd.DataFrame(topic_counts, columns=['Thema', 'Anzahl'])
    sns.barplot(x='Anzahl', y='Thema', data=topics_df, palette='rocket')
    plt.title('Top 20 Themen nach Autorenvielfalt',
              fontsize=16, fontweight='bold')
    plt.xlabel('Anzahl einzigartiger Autoren', fontsize=12)
    plt.ylabel('Thema', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig('general_topics.png')
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='num_of_papers', y='num_topics_covered',
                    data=df, alpha=0.6)
    plt.title('Anzahl Paper vs. Anzahl abgedeckter Themen', fontsize=16)

    plt.savefig('papers_vs_topics_scatter.png')
