from config import DATA_DIR, MODELS_DIR
import torch
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm
from safetensors.torch import save_file

from src.utils.llm import get_qwen_model


def extract_and_save_contrastive_style_activations(
    model,
    tokenizer,
    dataset_path: str,
    contrast_dataset_path: str,
    output_path: str,
    layers: list[int],
    dataset_split="test"
):
    """
    Lädt ein Test-Dataset und ein Contrast-Dataset.
    Berechnet für die 3 Examples pro Paper den Contrastive Style Vector
    (Ground Truth Abstract - Synthetic Abstract) aus dem REINEN Text 
    und speichert den Durchschnitt ab.
    """

    print(f"Lade Test-Dataset von: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    if dataset_split in dataset:
        test_dataset = dataset[dataset_split]
    else:
        test_dataset = dataset

    print(f"Lade Contrast Dataset von: {contrast_dataset_path}")
    contrast_dataset = DatasetDict.load_from_disk(contrast_dataset_path)
    contrast_dataset = concatenate_datasets(list(contrast_dataset.values()))

    print("Baue Nachschlage-Tabelle für Contrast-Daten...")
    contrast_data_dict = {}
    for row in contrast_dataset:
        p_id = str(row["paper_id"])  # type: ignore
        contrast_data_dict[p_id] = {
            "target": row["ground_truth"],  # type: ignore
            "source": row["gen_abstract"]     # type: ignore
        }

    activations_dict = {}

    def get_layer_activations(text: str):
        # Einfach nur den rohen Text tokenisieren, ohne jegliches Prompt-Format.
        # Truncation hinzugefügt, falls ein Abstract das Token-Limit sprengt.
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        return outputs.hidden_states

    for row in tqdm(test_dataset, desc="Berechne Contrastive Vectors"):
        paper_id = str(row["paper_id"])  # type: ignore

        example_ids = [
            str(row["example_1_id"]),  # type: ignore
            str(row["example_2_id"]),  # type: ignore
            str(row["example_3_id"])  # type: ignore
        ]

        diff_vectors_per_layer = {layer: [] for layer in layers}
        valid_examples_count = 0

        for ex_id in example_ids:
            if ex_id not in contrast_data_dict:
                print(
                    f"\nWarnung: Example ID {ex_id} nicht im Contrast Dataset gefunden.")
                continue

            target_text = contrast_data_dict[ex_id]["target"]
            source_text = contrast_data_dict[ex_id]["source"]

            target_states = get_layer_activations(target_text)
            source_states = get_layer_activations(source_text)

            # Differenz pro Layer berechnen
            for layer in layers:
                # Durchschnitt über alle Tokens des rohen Abstracts bilden
                target_mean = target_states[layer].mean(dim=1).squeeze(0)
                source_mean = source_states[layer].mean(dim=1).squeeze(0)

                # Der Contrastive Vector: Target - Source
                diff_vector = target_mean - source_mean
                diff_vectors_per_layer[layer].append(diff_vector)

            valid_examples_count += 1

        if valid_examples_count == 0:
            print(
                f"\nWarnung: Keine gültigen Examples für Paper {paper_id} gefunden. Überspringe Paper.")
            continue

        # 4. Durchschnitt der 3 Example-Vektoren pro Layer bilden und speichern
        for layer in layers:
            stacked_diffs = torch.stack(diff_vectors_per_layer[layer])
            final_style_vector = stacked_diffs.mean(dim=0).cpu().clone()

            dict_key = f"{paper_id}_layer_{layer}"
            activations_dict[dict_key] = final_style_vector

    print(
        f"\nSpeichere {len(activations_dict)} Contrastive Vektoren nach {output_path}...")
    save_file(activations_dict, output_path)
    print("Fertig!")


def extract_and_save_paper_approach_style_activations(
    model,
    tokenizer,
    dataset_path: str,
    contrast_dataset_path: str,
    output_path: str,
    layers: list[int],
    dataset_split="test"
):
    """
    Setzt den Activation-based Ansatz aus dem Paper um: 
    Style Vector = mean(Author's 3 Examples) - mean(All OTHER Author's Examples).
    Nutzt als Daten-Pool exakt die Examples, die im Test-Dataset definiert sind.
    """

    print(f"Lade Test-Dataset von: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    test_dataset = dataset[dataset_split] if dataset_split in dataset else dataset

    print(
        f"Lade Contrast Dataset (als Text-Lookup) von: {contrast_dataset_path}")
    contrast_dataset = DatasetDict.load_from_disk(contrast_dataset_path)
    contrast_dataset = concatenate_datasets(list(contrast_dataset.values()))

    # Nachschlagetabelle für die Texte (wir brauchen nur ground_truth)
    text_lookup = {}
    for row in contrast_dataset:
        p_id = str(row["paper_id"])  # type: ignore
        text_lookup[p_id] = row["ground_truth"]  # type: ignore

    def get_layer_activations(text: str):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        return outputs.hidden_states

    # -------------------------------------------------------------------------
    # PHASE 1: Sammle ALLE Examples aus dem Test-Dataset und berechne sie vor
    # -------------------------------------------------------------------------
    all_example_ids = set()
    for row in test_dataset:
        all_example_ids.add(str(row["example_1_id"]))  # type: ignore
        all_example_ids.add(str(row["example_2_id"]))  # type: ignore
        all_example_ids.add(str(row["example_3_id"]))  # type: ignore

    print(
        f"\nPhase 1: Precompute von {len(all_example_ids)} Examples aus dem Test-Set...")

    precomputed_activations = {}
    global_sums = {layer: 0 for layer in layers}

    for ex_id in tqdm(all_example_ids, desc="Precompute Activations"):
        if ex_id not in text_lookup:
            print(f"Warnung: Example {ex_id} fehlt im Lookup-Dataset!")
            continue

        text = text_lookup[ex_id]
        states = get_layer_activations(text)

        layer_means = {}
        for layer in layers:
            # Durchschnitt über alle Tokens des Abstracts
            mean_vec = states[layer].mean(dim=1).squeeze(0).cpu()
            layer_means[layer] = mean_vec

            # Globale Summe aufbauen
            global_sums[layer] = global_sums[layer] + mean_vec

        precomputed_activations[ex_id] = layer_means

    total_valid_examples = len(precomputed_activations)
    if total_valid_examples <= 3:
        raise ValueError(
            "Zu wenige gültige Examples für einen sinnvollen Kontrast gefunden.")

    # -------------------------------------------------------------------------
    # PHASE 2: Berechnung der Contrastive Style Vectors
    # -------------------------------------------------------------------------
    activations_dict = {}

    for row in tqdm(test_dataset, desc="Berechne Style Vectors"):
        paper_id = str(row["paper_id"])  # type: ignore

        author_ex_ids = [
            str(row["example_1_id"]),  # type: ignore
            str(row["example_2_id"]),  # type: ignore
            str(row["example_3_id"])  # type: ignore
        ]

        valid_author_ex_ids = [
            ex for ex in author_ex_ids if ex in precomputed_activations]

        if not valid_author_ex_ids:
            print(
                f"\nÜberspringe {paper_id} - keine gültigen Examples vorhanden.")
            continue

        num_author_ex = len(valid_author_ex_ids)
        num_others = total_valid_examples - num_author_ex

        if num_others <= 0:
            print(
                f"\nÜberspringe {paper_id} - keine 'anderen' Examples übrig.")
            continue

        for layer in layers:
            # 1. Target-Aktivierungen (a_s) für diesen Autor
            target_vecs = torch.stack(
                [precomputed_activations[ex][layer] for ex in valid_author_ex_ids])
            a_s = target_vecs.mean(dim=0)

            # 2. Andere-Aktivierungen (a_{S \setminus s}) effizient berechnen
            target_sum = target_vecs.sum(dim=0)
            other_sum = global_sums[layer] - target_sum
            a_S_minus_s = other_sum / num_others

            # 3. Contrastive Vector nach Paper-Formel
            style_vector = a_s - a_S_minus_s

            dict_key = f"{paper_id}_layer_{layer}"
            activations_dict[dict_key] = style_vector.clone()

    print(
        f"\nSpeichere {len(activations_dict)} Paper-Style-Vektoren nach {output_path}...")
    save_file(activations_dict, output_path)
    print("Fertig!")


if __name__ == "__main__":
    model, tokenizer = get_qwen_model()

    # Changing variables
    TARGET_LAYERS = [18, 19, 20]
    APPROACH = "own"  # "own" oder "paper"
    DATASET_SPLIT = "test"  # "train", "val" oder "test"

    # Pfade anpassen!
    DATASET_PATH = f"{DATA_DIR}/datasets/unseen_paper_dataset_icl"
    CONTRAST_DATASET_PATH = f"{DATA_DIR}/datasets/Qwen3_unseen_abstracts_dataset_full"
    OUTPUT_FILE = f"{DATA_DIR}/steering_vectors/unseen_{DATASET_SPLIT}_contrastive_{APPROACH}_{('_').join(map(str, TARGET_LAYERS))}.safetensors"

    if APPROACH == "own":
        extract_and_save_contrastive_style_activations(
            model=model,
            tokenizer=tokenizer,
            dataset_path=DATASET_PATH,
            contrast_dataset_path=CONTRAST_DATASET_PATH,
            output_path=OUTPUT_FILE,
            layers=TARGET_LAYERS,
            dataset_split=DATASET_SPLIT
        )
    elif APPROACH == "paper":
        extract_and_save_paper_approach_style_activations(
            model=model,
            tokenizer=tokenizer,
            dataset_path=DATASET_PATH,
            contrast_dataset_path=CONTRAST_DATASET_PATH,
            output_path=OUTPUT_FILE,
            layers=TARGET_LAYERS,
            dataset_split=DATASET_SPLIT
        )
