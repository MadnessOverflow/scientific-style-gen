from config import DATA_DIR, MODELS_DIR

# Fix für Titan RTX (Turing) - Deaktiviert xFormers,
# da Unsloth bei eingeschränkten target_modules auf Standard-PEFT zurückfällt.
try:
    import unsloth.utils.attention_dispatch as ad
    ad.HAS_XFORMERS = False
except ImportError:
    pass

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from transformers import EarlyStoppingCallback
from datasets import DatasetDict
from src.utils.prompts import ICL_SYSTEM_PROMPT, ICL_USER_PROMPT, LORA_SYSTEM_PROMPT, LORA_USER_PROMPT

DEFAULT_PARAMETERS = {
    "r": 8,
    "lora_alpha": 16,
    "lr": 2e-4,
    "label_smoothing_factor": 0.0,
}


def get_fine_tune_dataset(tokenizer):
    dataset = DatasetDict.load_from_disk(
        f"{DATA_DIR}/datasets/paper_dataset_with_summaries_unsloth_qwen3-4b-instruct-2507-unsloth-bnb-4bit")

    def formatting_prompts_func(examples):
        output_messages = []

        for i in range(len(examples['summary'])):
            user_content = LORA_USER_PROMPT.format(
                author=examples["author"][i], summary=examples["summary"][i])

            assistant_content = examples['ground_truth'][i]

            messages = [
                {"role": "system", "content": LORA_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]

            output_messages.append(messages)

        texts = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False,
                                               truncation=True, max_length=4096) for conversation in output_messages]

        return {"text": texts}

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        batch_size=32,
        # Entfernt die alten Spalten (example_1, etc.)
        remove_columns=dataset["train"].column_names
    )

    return dataset


def get_fine_tune_dataset_ICL(tokenizer):
    dataset = DatasetDict.load_from_disk(f"{DATA_DIR}/datasets/paper_dataset_icl")

    def formatting_prompts_func(examples):
        output_messages = []

        for i in range(len(examples['summary'])):
            abstract_examples = [examples['example_1'][i],
                                 examples['example_2'][i], examples['example_3'][i]]
            user_content = ICL_USER_PROMPT.format(
                examples="\n\n".join(
                    [f"Example {i+1}: \n{s}" for i, s in enumerate(abstract_examples)]),
                summary=examples["summary"][i]
            )

            assistant_content = examples['ground_truth'][i]

            messages = [
                {"role": "system", "content": ICL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]

            output_messages.append(messages)

        texts = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False,
                                               truncation=True, max_length=4096) for conversation in output_messages]

        return {"text": texts}

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        batch_size=32,
        # Entfernt die alten Spalten (example_1, etc.)
        remove_columns=dataset["train"].column_names
    )

    return dataset


def lora_fine_tune_model(base_model, tokenizer, dataset, model_name: str, new_parameters=None):
    """
    https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Instruct.ipynb#scrollTo=1ahE8Ys37JDJ
    """
    if new_parameters is None:
        new_parameters = {}
    model_parameters = DEFAULT_PARAMETERS | new_parameters

    print("Parameters:")
    print(f"{model_parameters}\n")

    model = FastLanguageModel.get_peft_model(
        base_model,
        r=model_parameters["r"],
        # target_modules = ["q_proj", "v_proj"],
        lora_alpha=model_parameters["lora_alpha"],
        use_gradient_checkpointing=False,  # type: ignore
        random_state=2507,

    )

    print(model)

    for name, module in model.named_modules():
        if "lora_A" in name:
            print(f"LoRA angewendet auf: {name}")

    dataset = dataset.shuffle(seed=42)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        args=SFTConfig(
            dataset_text_field="text",
            output_dir=f"{MODELS_DIR}/.temp",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=4,

            eval_strategy="steps",
            eval_steps=25,
            save_strategy="steps",
            save_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,

            warmup_steps=5,
            num_train_epochs=50,
            learning_rate=model_parameters["lr"],
            fp16=True,
            bf16=False,
            logging_steps=25,
            optim="adamw_torch",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,

            label_smoothing_factor=model_parameters["label_smoothing_factor"],
            neftune_noise_alpha=5.0
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=20,
                early_stopping_threshold=0.0
            )
        ]
    )

    # Make sure it only trains on responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("\nExample for a conversation with chat_template\nWITHOUT masking: ")
    print(tokenizer.decode(trainer.train_dataset[67]["input_ids"]))

    print("\nWITH masking: ")
    print(tokenizer.decode([tokenizer.pad_token_id if x == -
          100 else x for x in trainer.train_dataset[67]["labels"]]))
    print("\n")

    # 6. Training & Speichern
    trainer.train()

    print(
        f"Training beendet.\nDer beste eval_loss beträgt: {trainer.state.best_metric}")

    # Speichern für Inference (nur LoRA Adapter)
    model.save_pretrained(f"{MODELS_DIR}/{model_name}_lora")

    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(f"{MODELS_DIR}/{model_name}_lora")

    print(f"Model saved to models/{model_name}_lora")


def get_lora_model(model_id):
    model_name = model_id.replace('/', '_')

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"{MODELS_DIR}/{model_name}_lora",
        max_seq_length=4096,
        load_in_4bit=False,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    return model, tokenizer


if __name__ == "__main__":
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    icl = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=8192,
        load_in_4bit=False,
        dtype=None
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3-instruct",
    )

    if icl:
        dataset = get_fine_tune_dataset_ICL(tokenizer)
    else:
        dataset = get_fine_tune_dataset(tokenizer)

    model_name = model_id.replace(
        '/', '_') + ("_icl" if icl else "") + "_ceiling"
    lora_fine_tune_model(model, tokenizer, dataset, model_name)
