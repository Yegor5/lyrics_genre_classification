import argparse
import torch
import logging
import yaml

import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)


logger = logging.getLogger(__name__)


class PredictionProcessor:
    def __init__(self, model, tokenizer, label_names=None, threshold=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.label_names = label_names
        self.threshold = threshold

    def preproc_text(self, text, max_length):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Текст должен быть непустой строкой")

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        return encoding

    def raw_predict(self, text, max_length):
        inputs = self.preproc_text(text, max_length)
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.model.device)).logits
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        return probs

    def process_prediction(self, probs):
        try:
            if not isinstance(probs, (np.ndarray, list)):
                raise ValueError(f"Ожидается np.ndarray или list, получен {type(probs)}")

            probs = np.array(probs)

            if probs.ndim != 1:
                raise ValueError(f"Ожидается 1D массив, получен {probs.ndim}D")

            if np.any(probs < 0) or np.any(probs > 1):
                raise ValueError("Вероятности должны быть в диапазоне [0, 1]")
            
            if self.label_names and len(probs) != len(self.label_names):
                raise ValueError(f"Длина массива вероятностей ({len(probs)}) не совпадает с количеством классов")

            predicts = (probs >= self.threshold).astype(int)
            pred_indices = np.where(predicts == 1)[0].tolist()
            
            if len(pred_indices) == 0:
                pred_indices = [int(np.argmax(probs))]

            pred_labels, prob_labels = pred_indices, probs
            if self.label_names:
                pred_labels = [self.label_names[i] for i in pred_indices]
                prob_labels = {}
                for i, prob in enumerate(probs):
                    prob_labels[self.label_names[i]] = prob

            result = {
                "prob_labels": prob_labels,
                "pred_labels": pred_labels
            }

            return result

        except Exception as e:
            logger.error(f"Ошибка при обработке предсказания: {e}")
            raise

    def predict_single(self, text, max_length):
        probs = self.raw_predict(text, max_length)
        return self.process_prediction(probs)

    def predict_batch(self, texts, max_length):
        results = []
        for text in texts:
            try:
                result = self.predict_single(text, max_length)
                results.append(result)
            except Exception as e:
                logger.error(f"Ошибка при предсказании для текста '{text}': {e}")
                results.append({"error": str(e)})
        return results


def main(args):
    
    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded")
    
    df = pd.read_csv(args.input_path)
    if "lyrics" not in df.columns:
        raise ValueError("Ожидается колонка lyrics")
    logger.info("Data loaded")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["eval_params"]["model_path"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["eval_params"]["model_path"])
    model.to(device)
    model.eval()
    logger.info("Model loaded")

    label_names = [
        model.config.id2label[i]
        for i in range(len(model.config.id2label))
    ]

    processor = PredictionProcessor(
        model=model,
        tokenizer=tokenizer,
        label_names=label_names,
        threshold=cfg["eval_params"]["threshold"],
    )

    results = processor.predict_batch(
        df["lyrics"].tolist(),
        max_length=cfg["train_params"]["max_length"],
    )

    out_df = pd.DataFrame({
        "id": df.index,
        "pred_labels": [
            "|".join(r["pred_labels"]) if "pred_labels" in r else "error"
            for r in results
        ]
    })

    out_df.to_csv(args.output_path, index=False)
    logger.info("Predict saved to %s", args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="path to input csv")
    parser.add_argument("--output_path", required=True, help="path to output csv")
    parser.add_argument("--config_path", required=True, help="path to params.yaml")
    args = parser.parse_args()
    main(args)