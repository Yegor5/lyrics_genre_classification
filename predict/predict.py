import torch
import numpy as np
import logging


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
            outputs = self.model(**inputs).logits
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
