from config import DATA_DIR, MODELS_DIR
from collections import Counter
from typing import List
import numpy as np
import torch
import faiss
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score
from pathlib import Path

from src.author_classifier.author_classification import load_author_classification_dataset

device = torch.device("cuda")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_mean_embedding_func(tokenizer, model):
    def get_embeddings(batch):
        inputs = tokenizer(
            batch['abstract'],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)

        with torch.no_grad():
            model_output = model(**inputs)

        embeddings = mean_pooling(model_output, inputs['attention_mask'])
        # embeddings = model_output.last_hidden_state[:, 0]

        return {"embeddings": embeddings.cpu().numpy()}

    return get_embeddings


def get_star_embedding_func(tokenizer, model):
    def get_embeddings(batch):
        inputs = tokenizer(
            batch['abstract'],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)

        with torch.no_grad():
            style_embeddings = model(
                inputs.input_ids, attention_mask=inputs.attention_mask).pooler_output

        return {"embeddings": style_embeddings.cpu().numpy()}

    return get_embeddings


def predict_dataset_with_knn(dataset, index, author_mapping, model, tokenizer, k=3, cosine_similarity=False):
    print(f"Generating embeddings for {len(dataset)} test samples...")

    embedded_test_set = dataset.map(
        get_star_embedding_func(
            tokenizer, model) if model.name_or_path == "AIDA-UPM/star" else get_mean_embedding_func(tokenizer, model),
        batched=True,
        batch_size=32
    )

    query_embeddings = np.array(
        embedded_test_set['embeddings']).astype('float32')

    if cosine_similarity:
        faiss.normalize_L2(query_embeddings)

    # Search through all distances for MRR and Margin Ratio
    search_k = index.ntotal
    print(
        f"Searching index for {len(query_embeddings)} queries (eval_k={search_k}, voting_k={k})...")
    distances, indices = index.search(
        query_embeddings, search_k)  # type: ignore

    true_labels = dataset['labels']

    predicted_authors = []
    margin_ratios = []
    mrrs = []

    for query_idx, (dist_array, idx_array) in enumerate(zip(distances, indices)):
        true_author = true_labels[query_idx]
        neighbor_authors = [author_mapping[i] for i in idx_array]

        # --- Top-K Voting ---
        top_k_authors = neighbor_authors[:k]
        vote = Counter(top_k_authors).most_common(1)
        predicted_authors.append(vote[0][0])

        # --- Other metrics ---
        d_true_min = None
        d_false_min = None
        rank = None

        for rank_idx, (dist, author) in enumerate(zip(dist_array, neighbor_authors)):
            # cosine sim to distance
            current_dist = max(
                0.0, 1.0 - dist) if cosine_similarity else max(0.0, float(dist))

            if author == true_author:
                if d_true_min is None:
                    d_true_min = current_dist
                    rank = rank_idx + 1
            else:
                if d_false_min is None:
                    d_false_min = current_dist

            if d_true_min is not None and d_false_min is not None:
                break

        # MRR
        mrrs.append(1.0 / rank if rank is not None else 0.0)

        # Margin Ratio
        if d_true_min is not None and d_false_min is not None:
            denom = d_false_min if d_false_min > 1e-6 else 1e-6
            margin_ratios.append(d_true_min / denom)
        else:
            margin_ratios.append(float('inf'))

    return predicted_authors, margin_ratios, mrrs


def train_retrieval_classifier(train_dataset: Dataset, model_path: str, cosine_similarity=False, save=False, save_path=f"{MODELS_DIR}/retrieval_based_classification/knn_model"):
    if model_path == "AIDA-UPM/star":
        tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    embedded_dataset = train_dataset.map(
        get_star_embedding_func(
            tokenizer, model) if model_path == "AIDA-UPM/star" else get_mean_embedding_func(tokenizer, model),
        batched=True,
        batch_size=32
    )

    print("Creating FAISS index...")
    embeddings = np.array(embedded_dataset['embeddings']).astype('float32')
    dimension = embeddings.shape[1]

    if cosine_similarity is False:
        index = faiss.IndexFlatL2(dimension)
    else:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)  # type: ignore

    print(
        f"Index created with {index.ntotal} vectors of dimension {dimension}")

    # Speichere die Autoren in der *exakt gleichen Reihenfolge*
    # Pos 0 im Index -> Autor an Pos 0 in dieser Liste
    author_mapping = list(embedded_dataset['labels'])

    if save:
        save_path = Path(
            f"{save_path}_{"cos_sim" if cosine_similarity else "L2"}/")
        save_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, f"{save_path}/author_index.faiss")
        with open(f"{save_path}/author_mapping.json", "w") as f:
            json.dump(author_mapping, f)

        print("Index and author mapping saved successfully!")

    return index, author_mapping


def evaluate_retrieval_classifier(test_dataset: Dataset, model_path: str, index: faiss.IndexFlat, author_mapping: List[int], cosine_similarity: bool = False, k: int = 5):
    if model_path == "AIDA-UPM/star":
        tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    print(f"Ready for inference. Index has {index.ntotal} samples.")

    predictions, _, _ = predict_dataset_with_knn(
        test_dataset, index, author_mapping, model, tokenizer, k, cosine_similarity)

    # Accuracy
    acc = accuracy_score(
        test_dataset['labels'],
        predictions
    )
    print(f"\nGenauigkeit auf dem Test-Set (k={k}): {acc * 100:.4f}%")

    return acc


if __name__ == "__main__":
    # "StyleDistance/styledistance" # "AnnaWegmann/Style-Embedding" # f"{MODELS_DIR}/distilbert-base-uncased/model"
    MODEL_PATH = "AIDA-UPM/star"

    dataset = load_author_classification_dataset()
    # concatenate_datasets([dataset["train"], dataset["val"]])
    combined_train = dataset["train"]

    index, author_mapping = train_retrieval_classifier(
        combined_train, model_path=MODEL_PATH, cosine_similarity=True, save=False, save_path=f"{MODELS_DIR}/retrieval_based_classification/knn_model")
    evaluate_retrieval_classifier(dataset["val"], model_path=MODEL_PATH,
                                  index=index, author_mapping=author_mapping, k=3, cosine_similarity=True)
