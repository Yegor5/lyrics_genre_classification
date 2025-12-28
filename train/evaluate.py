import argparse
import json
import logging
import yaml
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from utils.data_preproc import preproc_dataset
from utils.compute_metrics import compute_metrics


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(config_path):
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded")

    split, num_labels = preproc_dataset(cfg)
    val_ds = split["test"]
    logger.info("Data loaded")

    model_path = cfg["save_params"]["save_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    ).to(device)
    logger.info("Model loaded")

    eval_args = TrainingArguments(
        output_dir=cfg["eval_params"]["metrics_path"],
        per_device_eval_batch_size=cfg["train_params"]["batch_size"],
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    logger.info("Evaluation metrics: %s", metrics)

    metrics_path = cfg["eval_params"]["metrics_path"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)