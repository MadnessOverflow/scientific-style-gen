import unsloth
from config import DATA_DIR, MODELS_DIR
import torch
import re
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict
from safetensors.torch import save_file
from transformers import set_seed

from src.utils.prompts import DEFAULT_ABSTRACT_SYSTEM_PROMPT, DEFAULT_ABSTRACT_USER_PROMPT
from src.utils.llm import get_qwen_model


def get_extraction_hook(layer_idx: int, storage_dict: dict):
    """
    Erzeugt eine Hook-Funktion, die die Hidden States während der 
    Inferenz abfängt und im Speicher ablegt.
    """
    def hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # Sofort vom Graphen trennen und auf die CPU schieben, um VRAM zu sparen
        storage_dict[layer_idx].append(hidden_states.detach().cpu())
        return output
    return hook


def extract_activations_during_generation(
    model,
    tokenizer,
    dataset_name: str,
    output_activations_path: str,
    target_layer_indices: list[int]
):
    """
    Lädt das Dataset, formatiert die Prompts, generiert die Abstracts und
    extrahiert gleichzeitig die durchschnittlichen Hidden Activations pro Paper.
    """
    dataset_path = f"{DATA_DIR}/datasets/{dataset_name}"

    print(f"Lade Test-Daten aus: {dataset_path}")
    loaded_data = load_from_disk(dataset_path)
    if isinstance(loaded_data, DatasetDict):
        if "test" in loaded_data:
            dataset = loaded_data["test"]
        else:
            print("Hinweis: Split 'test' nicht gefunden. Breche ab.")
            return
    else:
        dataset = loaded_data

    def format_inference_prompt(example):
        messages = [
            {"role": "system", "content": DEFAULT_ABSTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": DEFAULT_ABSTRACT_USER_PROMPT.format(
                summary=example["summary"])},
        ]
        return {
            "text_prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        }

    print("Formatiere Prompts...")
    dataset = dataset.map(format_inference_prompt)

    print(f"Starte Generierung und Extraktion für {len(dataset)} Einträge...")
    set_seed(67)

    generated_abstracts = []
    final_activations_dict = {}

    for row in tqdm(dataset, desc="Inference & Extraction"):
        paper_id = str(row["paper_id"])
        prompt = row["text_prompt"]

        inputs = tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True, max_length=4096
        ).to(model.device)

        input_ids = inputs["input_ids"]

        # Temporärer Speicher für die Hooks dieser spezifischen Generierung
        storage_dict = {layer: [] for layer in target_layer_indices}
        active_hooks = []

        # Hooks registrieren
        for layer_idx in target_layer_indices:
            target_layer = model.model.layers[layer_idx]
            hook_handle = target_layer.register_forward_hook(
                get_extraction_hook(layer_idx, storage_dict)
            )
            active_hooks.append(hook_handle)

        try:
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=1024,
                    do_sample=True,
                    use_cache=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0,
                    repetition_penalty=1.05,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            # Hooks sicherstellen entfernen, auch wenn Fehler auftreten
            for hook in active_hooks:
                hook.remove()

        # --- Text Verarbeitung ---
        input_length = input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        decoded_text = tokenizer.decode(
            generated_tokens, skip_special_tokens=True)
        cleaned_text = re.sub(r"\*\*abstract\*\*\s*", "",
                              decoded_text, flags=re.IGNORECASE).strip()
        generated_abstracts.append(cleaned_text)

        # --- Activations Verarbeitung ---
        for layer_idx in target_layer_indices:
            # Kombiniere Pre-fill Phase und alle Decoding Steps
            full_sequence_hiddens = torch.cat(storage_dict[layer_idx], dim=1)

            # Durchschnitt über die gesamte Sequenz
            mean_hidden = full_sequence_hiddens.mean(dim=1).squeeze(0)

            # Key Format: "12345_layer_12" (Sauber auf das Paper gemappt)
            dict_key = f"{paper_id}_layer_{layer_idx}"
            final_activations_dict[dict_key] = mean_hidden.clone()

    print("\nGenerierung beendet.")

    # --- Speichern der Safetensors ---
    print(
        f"Speichere {len(final_activations_dict)} Activation-Vektoren nach {output_activations_path}...")
    save_file(final_activations_dict, output_activations_path)
    print("Fertig!")


if __name__ == "__main__":
    model, tokenizer = get_qwen_model()

    extract_activations_during_generation(
        model=model,
        tokenizer=tokenizer,
        dataset_name="paper_dataset_icl",
        output_activations_path=f"{DATA_DIR}/steering_vectors/18_19_20_hidden_activations.safetensors",
        target_layer_indices=[18, 19, 20]
    )
