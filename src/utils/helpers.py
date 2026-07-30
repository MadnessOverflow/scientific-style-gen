import gc
import torch

def flush_vram():
    """Bereinigt den VRAM und Garbage Collector."""
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM erfolgreich bereinigt.\n")

def get_layer_activations(model, tokenizer, text: str, max_length: int = 4096):
    """
    Tokenizes raw text and extracts hidden states from the model.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs.hidden_states
