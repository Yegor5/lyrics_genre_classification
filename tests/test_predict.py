import torch
import pytest
import numpy as np
from predict.predict import PredictionProcessor


class DummyModel:
    def __call__(self, **inputs):
        class Output:
            logits = torch.tensor([[0.2, 1.5, -0.3]])
        return Output()


class DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}


@pytest.fixture
def processor():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    label_names = ["pop", "rock", "jazz"]
    return PredictionProcessor(model, tokenizer, label_names, threshold=0.5)


def test_process_prediction_basic(processor):
    probs = np.array([0.8, 0.2, 0.6])
    result = processor.process_prediction(probs)
    assert set(result["pred_labels"]) == {"pop", "jazz"}
    assert "rock" in result["prob_labels"]


def test_process_prediction_invalid_type(processor):
    with pytest.raises(TypeError):
        processor.process_prediction("not-an-array")


def test_process_prediction_invalid_shape(processor):
    with pytest.raises(ValueError):
        processor.process_prediction([[0.1, 0.2]])


def test_process_prediction_invalid_range(processor):
    with pytest.raises(ValueError):
        processor.process_prediction([1.2, -0.1, 0.5])


def test_process_prediction_label_mismatch(processor):
    probs = [0.5, 0.7]
    with pytest.raises(ValueError):
        processor.process_prediction(probs)


def test_preprocess_text_validation(processor):
    with pytest.raises(ValueError):
        processor.preproc_text("", max_length=10)


def test_predict_batch_with_error(processor):
    results = processor.predict_batch(["hi", ""], max_length=10)
    assert "error" in results[1]
