from config import DATA_DIR, MODELS_DIR
from unsloth import unsloth_train
import os
import torch
from typing import List, Dict, Any, Optional
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from src.utils.prompts import DEFAULT_ABSTRACT_SYSTEM_PROMPT, DEFAULT_ABSTRACT_USER_PROMPT
from src.style_control_network.implementation.end_to_end_model import EndToEndSteeredLLM


class SteeredDataCollator:
    """
    Ein eigener Collator, um Tensoren normal zu verarbeiten, aber die
    'author_abstracts' als reine Python-Listen an unser Custom Model weiterzuleiten.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Iterieren über alle Samples im Batch
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long)
                     for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long)
                  for f in features]
        author_abstracts = [f["author_abstracts"] for f in features]

        # RNN Padding Logik
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # Batching Text Sequences (Padding to longest in batch)
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id)
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100)

        # Attention Mask berechnen (1 für Tokens, 0 für Padding)
        attention_mask = (batch_input_ids != pad_token_id).long()

        return {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask,
            "labels": batch_labels,
            "author_abstracts": author_abstracts
        }


class HypernetworkTrainer(Trainer):
    """
    Angepasster Trainer, der ausschließlich die Gewichte des Hypernetworks 
    in den Checkpoints abspeichert (spart gigantische Mengen von VRAM / Festplattenspeicher).
    """

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if output_dir:
            self.model.save(output_dir)  # type: ignore
        else:
            print(
                "Warnung: Kein output_dir angegeben, Hypernetwork wird nicht gespeichert.")

    def _load_best_model(self):
        """
        Wird am Ende des Trainings aufgerufen, wenn load_best_model_at_end=True gesetzt ist.
        Verhindert, dass der HF Trainer nach Standard-Dateien (wie pytorch_model.bin) sucht.
        """
        if self.state.best_model_checkpoint is not None:
            best_model_path = os.path.join(
                self.state.best_model_checkpoint, "style_control_network.pt")
            if os.path.exists(best_model_path):
                print(
                    f"Lade das beste Modell aus: {best_model_path} (score: {self.state.best_metric})")
                self.model.load(best_model_path)  # type: ignore
            else:
                print(f"Warnung: {best_model_path} nicht gefunden!")
        else:
            print(
                "Warnung: Kein best_model_checkpoint gefunden, Modell kann nicht geladen werden.")


def get_style_control_network_dataset(tokenizer, dataset_path: str):
    dataset = DatasetDict.load_from_disk(dataset_path)

    def formatting_func(examples):
        author_abstracts_list = []
        prompts = []
        responses = []

        # Über den Block (Beispiele) iterieren
        for i in range(len(examples['summary'])):
            # Die 3 relevanten Beispiele holen (in der Zukunft für den Embedding Model Aufruf benötigt)
            abstracts = [
                examples['example_1'][i],
                examples['example_2'][i],
                examples['example_3'][i]
            ]
            author_abstracts_list.append(abstracts)

            # Text formulieren
            system_msg = {"role": "system",
                          "content": DEFAULT_ABSTRACT_SYSTEM_PROMPT}
            user_msg = {"role": "user", "content": DEFAULT_ABSTRACT_USER_PROMPT.format(
                summary=examples['summary'][i])}

            conversation_prompt = [system_msg, user_msg]

            prompt_str = tokenizer.apply_chat_template(
                conversation_prompt, tokenize=False, add_generation_prompt=True)
            response_str = examples['ground_truth'][i] + "<|im_end|>\n"

            prompts.append(prompt_str)
            responses.append(response_str)

        tokenized_seq = tokenizer(
            prompts,
            responses,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            max_length=4096,
        )

        input_ids_list = tokenized_seq["input_ids"]
        labels_list = []

        for i in range(len(input_ids_list)):
            sequence_ids = tokenized_seq.sequence_ids(i)
            # sequence_id == 0 -> entspricht dem prompt (alles wird auf -100 ignoriert)
            # sequence_id == 1 -> entspricht der Response (Label wird behalten)
            labels = [-100 if seq_id != 1 else label for seq_id,
                      label in zip(sequence_ids, input_ids_list[i])]
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "author_abstracts": author_abstracts_list
        }

    dataset = dataset.map(
        formatting_func,
        batched=True,
        batch_size=32,
        remove_columns=dataset["train"].column_names
    )

    return dataset


if __name__ == "__main__":
    model_name = "hyp_lora_3_lay"

    dataset_path = f"{DATA_DIR}/datasets/paper_dataset_icl"
    model_output_dir = f"{MODELS_DIR}/style_control_network/{model_name}"

    print("Initialisiere End-to-End Modell...")

    model = EndToEndSteeredLLM(
        target_layers=[18, 19, 20],  # list(range(36)), #
        alpha=1.0,
        neftune_alpha=5.0,
        steering_method="lora_l",
        style_control_network_kwargs={
            "embedder_name": "AIDA-UPM/star",
            # "target_modules": ["q_proj", "v_proj"],
            # "target_modules": [
            #     "q_proj", "k_proj", "v_proj", "o_proj",
            #     "gate_proj", "up_proj", "down_proj",
            # ]
        }
    )

    print(f"Lade Datenset von {dataset_path}...")
    dataset = get_style_control_network_dataset(model.tokenizer, dataset_path)

    dataset = dataset.shuffle(seed=42)

    print("Starte Trainer-Setup...")
    trainer = HypernetworkTrainer(
        model=model,
        processing_class=model.tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=SteeredDataCollator(model.tokenizer),
        args=TrainingArguments(
            output_dir=f"{MODELS_DIR}/.temp_hyper/{model_name}",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=2,

            eval_strategy="steps",
            eval_steps=25,
            save_strategy="steps",
            save_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,

            disable_tqdm=True,

            warmup_steps=5,
            num_train_epochs=50,
            learning_rate=2e-4,
            fp16=True,
            bf16=False,
            logging_steps=25,

            # Speicheroptimierungen
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            optim="adamw_torch",

            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            # Extrem wichtig, damit author_abstracts durch den Trainer läuft!
            remove_unused_columns=False,
            label_smoothing_factor=0.0,
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=25,
                early_stopping_threshold=0.0
            )
        ]
    )

    # Check
    print("\nSanity Check: Training Element - erste Zeile Decode")
    if trainer.train_dataset:
        train_dataset_first = trainer.train_dataset[0]
    else:
        raise ValueError("Trainingsdatensatz ist leer oder nicht vorhanden.")
    print(model.tokenizer.decode(train_dataset_first["input_ids"][:50]))
    print("Labels vorhanden (Maskiert -100): ",
          [-100 if x == -100 else x for x in train_dataset_first["labels"]][:50])

    print("Evaluation before training:")
    print(trainer.evaluate())

    print("\nBeginne Training...")
    trainer_stats = unsloth_train(trainer)
    print(f"Training abgeschlossen\nTrainer Stats: \n{trainer_stats}")

    print(f"Der beste eval_loss beträgt: {trainer.state.best_metric}")

    model.save(model_output_dir)
    print(f"Style Control Network Weights gespeichert in: {model_output_dir}")
