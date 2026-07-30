from config import DATA_DIR, MODELS_DIR
import json
from sklearn.model_selection import ParameterGrid
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import datetime
import sys

from src.author_classifier.retrieval_based_classification import evaluate_retrieval_classifier, train_retrieval_classifier
from src.author_classifier.author_classification import load_author_classification_dataset, train_aut_cls_model, DEFAULT_PARAM

if __name__ == "__main__":
    log_file_name = f"{MODELS_DIR}/hyperparameter_tuning/hyperparameter_tuning.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file_name, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    start_time = datetime.datetime.now()
    logging.info(
        f"Skript gestartet am: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Log-Datei wird nach '{log_file_name}' geschrieben.")
    logging.info("="*40)

    dataset = load_author_classification_dataset()

    label_names = dataset['train'].features['labels'].names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    logging.info(f"Daten geladen: {len(label_names)} Autoren gefunden.")
    logging.info(
        f"Samples: Train={len(dataset['train'])}, Val={len(dataset['val'])}, Test={len(dataset['test'])}")

    dataset.pop("test", None)
    logging.info(
        f"Test Split entfernt um Data Leakage zu vermeiden: {dataset}")

    grid_search_parameters = {
        # ['bert-base-uncased', 'bert-base-multilingual-cased', 'distilbert-base-uncased', 'distilbert-base-multilingual-cased'], # 'StyleDistance/styledistance'
        'model_name': ['StyleDistance/styledistance', 'AnnaWegmann/Style-Embedding', 'AIDA-UPM/star'],
        'retrieval_version': [True]  # [False, True, "fine-tuned"]
    }
    learning_rates = [5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 1e-6]
    param_grid = ParameterGrid(grid_search_parameters)

    all_results = []
    best_score = -1.0
    best_params = None
    best_model_path = "./models/hyperparameter_tuning/best_model"

    logging.info(f"Starte Grid Search mit {len(param_grid)} Kombinationen...")
    logging.info(f"Default Hyperparameter-Vorlage: {DEFAULT_PARAM}")
    for i, params in enumerate(param_grid):
        logging.info("")
        logging.info(f"--- Durchlauf {i+1}/{len(param_grid)} ---")
        logging.info(f"Parameter: {params}")

        model_name = params.pop('model_name', 'bert-base-multilingual-cased')
        metrics = {}

        if params.get('retrieval_version', False) is False:
            # aus parametern entfernen wegen trainer
            params.pop("retrieval_version", False)
            logging.info(f"Fine-tune model {model_name}")

            logging.info(f"Starte Suche nach der besten Learning Rate")
            # Unterschiedliche learning raten testen
            best_lr_loss = float('inf')
            best_lr_acc = 0
            best_lr = 0
            best_metrics = {}

            for lr in learning_rates:
                params["learning_rate"] = lr

                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(label_names),
                    id2label=id2label,
                    label2id=label2id
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                trainer, metrics = train_aut_cls_model(
                    model,
                    tokenizer,
                    dataset,
                    hyperparameters=params
                )

                if metrics.get('eval_loss', 0.0) < best_lr_loss:
                    best_lr_loss = metrics.get('eval_loss', 0.0)
                    best_lr_acc = metrics.get('eval_accuracy', 0.0)
                    best_lr = lr
                    best_metrics = metrics.copy()

                    output_dir = f"{MODELS_DIR}/hyperparameter_tuning/.temp/{model_name}"
                    trainer.save_model(
                        f"{MODELS_DIR}/hyperparameter_tuning/.temp/{model_name}")
                    tokenizer.save_pretrained(
                        f"{MODELS_DIR}/hyperparameter_tuning/.temp/{model_name}")

                logging.info(
                    f"Validation accuracy mit lr={lr}: {metrics.get('eval_accuracy', 0.0)} | Trainiert für {metrics.get('epochs', 0)} Epochen | Train loss: {metrics.get('train_loss')}, Eval loss: {metrics.get('eval_loss')}")

            params['learning_rate'] = best_lr
            # Zurück in die Parameter schreiben
            params['retrieval_version'] = False
            metrics = best_metrics

        else:
            model_path = None
            if params.get('retrieval_version', True) == "fine-tuned":
                model_path = f"{MODELS_DIR}/hyperparameter_tuning/.temp/{model_name}"
                logging.info(
                    f"Trainiere retrieval based classifier mit model fine-tuned-{model_name} ({model_path})")
            else:
                model_path = model_name
                logging.info(
                    f"Trainiere retrieval based classifier mit model {model_path}")

            # Evaluate for cos_sim oder l2 version
            l2_index, l2_author_mapping = train_retrieval_classifier(
                dataset["train"], model_path, cosine_similarity=False)
            cos_sim_index, cos_sim_author_mapping = train_retrieval_classifier(
                dataset["train"], model_path, cosine_similarity=True)

            # Evaluate for different k and distance metrics
            logging.info(
                "Starte Suche nach bestem k Wert und bester Distanz-Metrik.")
            best_score = -1.0
            best_k = 1
            best_cos_sim = False

            for k in range(1, 11, 2):
                score_l2 = evaluate_retrieval_classifier(
                    dataset["val"], model_path, l2_index, l2_author_mapping, cosine_similarity=False, k=k)
                score_cos_sim = evaluate_retrieval_classifier(
                    dataset["val"], model_path, cos_sim_index, cos_sim_author_mapping, cosine_similarity=True, k=k)

                logging.info(f"Validation accuracy mit k={k}, L2: {score_l2}")
                logging.info(
                    f"Validation accuracy mit k={k}, Cosine Sim: {score_cos_sim}")

                if score_l2 > best_score:
                    best_score = score_l2
                    best_k = k
                    best_cos_sim = False

                if score_cos_sim > best_score:
                    best_score = score_cos_sim
                    best_k = k
                    best_cos_sim = True

            logging.info(
                f"Choosing k={best_k} and {'Cosine similarity' if best_cos_sim else 'L2'} with score {best_score}")

            params['k'] = best_k
            params['cosine_similarity'] = best_cos_sim
            metrics = {'eval_accuracy': best_score}

        params['model_name'] = model_name

        current_score = metrics.get('eval_accuracy', 0.0)
        all_results.append({'params': params, 'metrics': metrics})
        logging.info("")
        logging.info("Statistiken des besten Modells:")

        if metrics.get('train_loss') is not None:
            logging.info(f"Trainiert für {metrics.get('epochs')} Epochen")
            logging.info(
                f"Train loss: {metrics.get('train_loss')} | Eval loss: {metrics.get('eval_loss')}")
        logging.info(f"Finale Accuracy (eval_accuracy): {current_score}")

        if current_score > best_score:
            best_score = current_score
            best_params = params
            best_params['model_name'] = model_name
            logging.info(f"Neues bestes Modell gefunden!")
            # trainer.save_model(best_model_path)

    logging.info("\n" + "="*40)
    logging.info("Grid Search abgeschlossen.")
    logging.info(
        f"Zusammenfassung aller Durchläufe (all_results): {all_results}\n")
    with open(f"{MODELS_DIR}/hyperparameter_tuning/all_runs.json", "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    logging.info(f"Bestes Modell gespeichert unter: {best_model_path}")
    logging.info(f"Beste Parameter: {best_params}")
    logging.info(f"Beste eval_accuracy (auf Validierungs-Set): {best_score}")
    logging.info("="*40 + "\n")

    # logging.info("Starte finale Evaluation mit dem besten Modell...")

    # tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    # model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    # testing_dataset = dataset["test"].map(get_tokenize_function(tokenizer), batched=True)

    # evaluation_results = evaluate_model(model, testing_dataset)

    # logging.info("\n" + "="*40)
    # logging.info("Finale Evaluations-Ergebnisse (auf Test-Set):")
    # logging.info(evaluation_results)
    # logging.info("="*40)

    # end_time = datetime.datetime.now()
    # logging.info(f"\nSkript beendet am: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # logging.info(f"Gesamtdauer: {end_time - start_time}")
