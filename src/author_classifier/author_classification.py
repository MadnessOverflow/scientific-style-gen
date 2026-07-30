from config import DATA_DIR, MODELS_DIR
from pathlib import Path
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import DatasetDict
import logging
import datetime
import sys


def load_author_classification_dataset():
    dataset = DatasetDict.load_from_disk(f'{DATA_DIR}/datasets/paper_dataset')
    dataset = dataset.class_encode_column('author')
    dataset = dataset.rename_column("author", "labels")

    return dataset


DEFAULT_PARAM = {
    "output_dir": f"{MODELS_DIR}/.temp/checkpoints",
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_strategy": "epoch",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 100,
    "optim": "adamw_torch",
    "weight_decay": 0.01,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.1
    # "label_names": ['author']
}


def get_tokenize_function(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['abstract'], padding="max_length", truncation=True)

    return tokenize_function


def train_aut_cls_model(model, tokenizer, dataset: DatasetDict, hyperparameters={}):
    training_hyperparameters = DEFAULT_PARAM.copy()
    training_hyperparameters.update(hyperparameters)

    training_dataset = dataset.map(
        get_tokenize_function(tokenizer), batched=True)

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = acc_metric.compute(predictions=predictions, references=labels)

        f1 = f1_metric.compute(predictions=predictions,
                               references=labels, average="weighted")

        return {**acc, **f1}  # type: ignore

    training_args = TrainingArguments(
        **training_hyperparameters
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset["train"],
        eval_dataset=training_dataset["val"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,  # type: ignore
        callbacks=[early_stopping_callback]
    )

    # logging.info("Starte das Training...")
    trainer.train()

    final_eval_metrics = trainer.evaluate()

    # Hole den Train loss des besten Modells
    best_metric_value = trainer.state.best_metric
    best_eval_entry_index = -1

    for i, log_entry in enumerate(trainer.state.log_history):
        if log_entry.get("eval_loss") == best_metric_value:
            best_eval_entry_index = i
            final_eval_metrics['epochs'] = int(log_entry.get('epoch', -1.0))
            break

    if best_eval_entry_index > 0:
        final_eval_metrics['train_loss'] = trainer.state.log_history[best_eval_entry_index - 1]['loss']

    return trainer, final_eval_metrics


def evaluate_model(model, dataset):
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = acc_metric.compute(predictions=predictions, references=labels)

        f1 = f1_metric.compute(predictions=predictions,
                               references=labels, average="macro")

        return {**acc, **f1}  # type: ignore

    training_args = TrainingArguments(
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,  # type: ignore
    )

    return trainer.evaluate()


if __name__ == "__main__":
    MODEL_NAME = 'distilbert-base-uncased'
    OUTPUT_DIR = f"{MODELS_DIR}/{MODEL_NAME.replace('/', '_')}/model"
    LOG_FILE_NAME = f"{MODELS_DIR}/{MODEL_NAME.replace('/', '_')}/training_eval.log"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        "learning_rate": 2e-05,
        "logging_dir": f"{MODELS_DIR}/{MODEL_NAME.replace('/', '_')}/logs",
        "report_to": "tensorboard"
    }

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE_NAME, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    start_time = datetime.datetime.now()
    logging.info(
        f"Skript gestartet am: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"MODELL: {MODEL_NAME}")

    training_hyperparameters = DEFAULT_PARAM.copy()
    training_hyperparameters.update(hyperparameters)
    logging.info(f"HYPERPARAMETER: {training_hyperparameters}")
    logging.info(f"AUSGABE-ORDNER: {OUTPUT_DIR}")
    logging.info(f"LOG-DATEI: {LOG_FILE_NAME}")
    logging.info("="*40)

    dataset = load_author_classification_dataset()

    label_names = dataset['train'].features['labels'].names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    num_labels = len(label_names)

    logging.info(f"Daten geladen: {num_labels} Autoren gefunden.")
    logging.info(f"Train/Validation-Split erstellt.")
    logging.info(
        f"Samples: Train={len(dataset['train'])}, Val={len(dataset['val'])}, Test={len(dataset['test'])}")

    logging.info(f"Lade Modell: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    trainer, metrics = train_aut_cls_model(
        model,
        tokenizer,
        dataset,
        hyperparameters=hyperparameters
    )

    logging.info("Training abgeschlossen.")
    logging.info(f"Validierungs-Metriken: {metrics}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info(f"Modell in {OUTPUT_DIR} gespeichert.")

    logging.info("\n" + "="*40)
    logging.info(f"Starte finale Evaluation mit Modell aus: {OUTPUT_DIR}")

    logging.info("Tokenisiere Test-Set...")
    testing_dataset = dataset["test"].map(
        get_tokenize_function(tokenizer), batched=True)

    evaluation_results = evaluate_model(model, testing_dataset)

    logging.info("\n" + "="*40)
    logging.info("Finale Evaluations-Ergebnisse (auf Test-Set):")
    logging.info(evaluation_results)
    logging.info("="*40)

    end_time = datetime.datetime.now()
    logging.info(
        f"\nSkript beendet am: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Gesamtdauer: {end_time - start_time}")
