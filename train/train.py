import argparse
import logging
import yaml
import torch
import subprocess
import mlflow
import mlflow.transformers
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from utils.data_preproc import preproc_dataset
from utils.compute_metrics import compute_metrics
from utils.callbacks import LogMetricsCallback


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_dvc_hash():
    try:
        return subprocess.check_output(
            ["dvc", "status", "-c"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main(config_path):

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded")
    
    mlflow.set_experiment("lyrics_genre_classification")
    mlflow.transformers.autolog(log_models=False)

    with mlflow.start_run():
        
        mlflow.log_params({
            "model_name": cfg["train_params"]["model_name"],
            "batch_size": cfg["train_params"]["batch_size"],
            "learning_rate": cfg["train_params"]["learning_rate"],
            "epochs": cfg["train_params"]["epochs"],
            "weight_decay": cfg["train_params"]["weight_decay"],
            "accum_size": cfg["train_params"]["accum_size"],
            "train_seed": cfg["train_params"]["seed"],
            "data_size": cfg["data_params"]["data_size"],
            "test_size": cfg["data_params"]["test_size"],
            "data_seed": cfg["data_params"]["seed"],
        })
        
        split, label_names = preproc_dataset(cfg)
        train_ds = split["train"]
        val_ds = split["test"]
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {label: i for i, label in enumerate(label_names)}
        logger.info("Data loaded")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(cfg["train_params"]["model_name"])
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["train_params"]["model_name"],
            num_labels=len(label_names),
            problem_type="multi_label_classification",
            id2label=id2label,
            label2id=label2id
        ).to(device)
        logger.info("Model loaded")

        training_args = TrainingArguments(
            output_dir=cfg["save_params"]["save_path"],
            eval_strategy="steps",
            eval_steps=cfg["eval_params"]["eval_steps"],
            save_strategy="steps",
            save_steps=cfg["save_params"]["save_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            learning_rate=cfg["train_params"]["learning_rate"],
            per_device_train_batch_size=cfg["train_params"]["batch_size"],
            per_device_eval_batch_size=cfg["train_params"]["batch_size"],
            gradient_accumulation_steps=cfg["train_params"]["accum_size"],
            fp16=True,
            num_train_epochs=cfg["train_params"]["epochs"],
            weight_decay=cfg["train_params"]["weight_decay"],
            logging_steps=cfg["log_params"]["log_steps"],
            logging_dir=cfg["log_params"]["log_path"],
            report_to="mlflow",
            seed=cfg["train_params"]["seed"]
        )

        callback = LogMetricsCallback(log_every=cfg["log_params"]["log_steps"], test_size=cfg["log_params"]["test_size"])
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[callback],
        )
        callback.trainer = trainer

        logger.info("Start training model")
        trainer.train()
        logger.info("Training finished")
        
        trainer.save_model(cfg["save_params"]["save_path"])
        mlflow.log_artifacts(cfg["save_params"]["save_path"], artifact_path="model")
        if Path("dvc.lock").exists():
            mlflow.log_artifact("dvc.lock")
        mlflow.set_tag("dvc_data_hash", get_dvc_hash())
        
        if cfg["save_params"]["save_hf"]:
            model.push_to_hub(cfg["save_params"]["hf_path"])
            tokenizer.push_to_hub(cfg["save_params"]["hf_path"])
            logger.info("Model saved on hf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to params.yaml")
    parser.add_argument("--verbose", action="store_true", help="enable verbose logging")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.INFO)
        logging.getLogger("transformers").setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

    main(args.config)
