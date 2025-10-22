import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (1 / (1 + np.exp(-logits))) > 0.5
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='micro', zero_division=0)
    recall = recall_score(labels, preds, average='micro', zero_division=0)
    f1 = f1_score(labels, preds, average='micro', zero_division=0)
    return {"accuracy": accuracy, "precision_micro": precision, "recall_micro": recall, "f1_micro": f1}
