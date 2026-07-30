from unsloth import FastLanguageModel

import gc
import time
from typing import List, Tuple
import torch

from src.utils.prompts import SUMMARY_USER_PROMPT, SUMMARY_SYSTEM_PROMPT

SAFE_CONTEXT_LIMIT = 7750


def create_chunks(text, tokenizer, overlap=250):
    """Zerlegt den Text intelligent anhand von Tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)
    print(f"Gesamtlänge des Papers: {total_tokens} Token")

    chunks = []
    chunk_lengths = []
    start = 0
    while start < total_tokens:
        end = min(start + SAFE_CONTEXT_LIMIT, total_tokens)
        chunk_ids = tokens[start:end]
        chunk_lengths.append(len(chunk_ids))
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

        if end == total_tokens:
            break

        start += SAFE_CONTEXT_LIMIT - overlap

    print(f"Text in {len(chunks)} Chunks zerlegt.")
    print(
        f"Chunks haben die folgenden Längen: [{", ".join([str(l) for l in chunk_lengths])}]")
    return chunks


def recursive_reduce(texts: List[str], model, tokenizer):
    """
    Fasst eine Liste von Texten rekursiv zusammen, bis sie klein genug ist um von einem LLM verarbeitet werden zu können.
    """
    current_texts = texts

    while len(current_texts) > 1:
        combined_text = "\n\n".join(
            [f"Section {i+1} Text:\n{s}" for i, s in enumerate(current_texts)])
        total_tokens = len(tokenizer.encode(
            combined_text, add_special_tokens=False))

        print(f"Aktuelle Token-Anzahl für Reduce: {total_tokens}")

        if total_tokens <= SAFE_CONTEXT_LIMIT:
            print(">>> Passt in den Kontext!")
            break

        print(
            f">>> Zu lang (> {SAFE_CONTEXT_LIMIT}). Starte Zusammenfassung...")

        BATCH_SIZE = 3
        summaries = []

        for i in range(0, len(current_texts), BATCH_SIZE):
            batch = current_texts[i: i + BATCH_SIZE]
            information = "\n\n".join(
                [f"Section {i+1} Text:\n{s}" for i, s in enumerate(batch)])

            print(f"   Verarbeite Batch {i//BATCH_SIZE + 1}...")
            task = "The following are summaries of a scientific paper's individual sections:\n\n"
            prompt = SUMMARY_USER_PROMPT.format(
                summaries=information, task=task)

            summary = parse_harmony_output(generate_text(
                model, tokenizer, SUMMARY_SYSTEM_PROMPT, prompt, 1024))
            summaries.append(summary)
            torch.cuda.empty_cache()
            gc.collect()

        current_texts = summaries
        print(
            f"Zwischenrunde fertig. Reduziert von {len(texts)} auf {len(current_texts)} Texte.")

    return current_texts


def generate_text(model, tokenizer, system_prompt, prompt, max_tokens, do_sample=True, temperature=1.0):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        # reasoning_effort="low"
    ).to(model.device)

    gen_start_time = time.time()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            use_cache=True,
            temperature=temperature, top_p=1.0, top_k=0,  # OpenAI oss settings
            # temperature=0.7,
            # top_p=0.8,
            # top_k=20,
            # min_p=0
            eos_token_id=tokenizer.eos_token_id
        )

    gen_end_time = time.time()
    print(f"Inferenz abgeschlossen in {gen_end_time - gen_start_time:.2f}s.")

    prompt_length = inputs.input_ids.shape[1]
    output_ids = generated_ids[0][prompt_length:]
    # Ohne skip kommt harmony format
    response_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return response_text


def split_thinking_output(text) -> Tuple[str, str | None]:
    separator = "assistantfinal"

    if separator in text:
        output = text.split(separator, 1)
        return output[1], output[0].removeprefix("analysis")

    return text, None


def parse_harmony_output(text):
    """
    Für OpenAI Modell den thinking part wegschneiden. (scheinbar einfach bei 'assistantfinal' splitten)
    """
    return split_thinking_output(text)[0].strip()


def get_gpt_model():
    model_id = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"

    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     dtype="auto",
    #     attn_implementation="eager"
    # )

    # return model, tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def get_qwen_model(load_in_4bit=False):
    model_id = "unsloth/Qwen3-4B-Instruct-2507"

    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     dtype="auto",
    #     attn_implementation="sdpa"
    # )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=8192,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def get_qwen_foundation_model():
    model_id = "Qwen/Qwen3-14B-Base"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer
