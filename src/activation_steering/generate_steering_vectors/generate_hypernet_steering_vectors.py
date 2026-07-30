from config import DATA_DIR, MODELS_DIR
import torch
from datasets import load_from_disk
from tqdm import tqdm
from safetensors.torch import save_file

from src.style_control_network.implementation.end_to_end_model import EndToEndSteeredLLM


def extract_and_save_style_control_networks_activations(
    model: EndToEndSteeredLLM,
    dataset_path: str,
    output_path: str,
):
    print(f"Lade Dataset von: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    if "test" in dataset:
        dataset = dataset["test"]

    activations_dict = {}

    for row in tqdm(dataset, desc="Generiere Hypernet Vektoren"):
        paper_id = str(row["paper_id"])  # type: ignore

        abstracts = [
            row["example_1"],  # type: ignore
            row["example_2"],  # type: ignore
            row["example_3"]  # type: ignore
        ]

        # Generiere Vectors (batch of 1)
        with torch.no_grad():
            steering_dict = model.style_control_network([abstracts])

        for layer, vec in steering_dict.items():
            # vec shape: [1, hidden_size] -> squeeze
            final_vec = vec.squeeze(0).cpu().clone()

            dict_key = f"{paper_id}_layer_{layer}"
            activations_dict[dict_key] = final_vec

    print(
        f"\nSpeichere {len(activations_dict)} Hypernet Steering Vektoren nach {output_path}...")
    save_file(activations_dict, output_path)
    print("Fertig!")


if __name__ == "__main__":
    # Setup
    model = EndToEndSteeredLLM(
        target_layers=[18, 19, 20],
        alpha=1.0,
        pretrained_style_control_network_path=f"{MODELS_DIR}/style_control_network/hyp_steering_star_embed/style_control_network.pt",
        steering_method="activation_steering",
        style_control_network_kwargs={
            "embedder_name": "AIDA-UPM/star",
        }
    )

    DATASET_PATH = f"{DATA_DIR}/datasets/paper_dataset_icl"
    OUTPUT_FILE = f"{DATA_DIR}/steering_vectors/test_style_control_network_steering_3_lay.safetensors"

    extract_and_save_style_control_networks_activations(
        model=model,
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_FILE,
    )
