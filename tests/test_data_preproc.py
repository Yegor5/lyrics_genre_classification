import pytest
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from train.utils.data_preproc import (
    check_data_train,
    check_data_structure,
    check_data_types,
    encode_dataset,
    tokenize_batch,
    convert_labels,
)


@pytest.fixture
def valid_dataset():
    data = Dataset.from_dict({
        "lyrics": ["song text 1", "song text 2"],
        "genre": [["pop"], ["rock"]]
    })
    return data


@pytest.fixture
def valid_dataset_train(valid_dataset):
    return DatasetDict({"train": valid_dataset})


def test_check_data_train_pass(valid_dataset_train):
    assert check_data_train(valid_dataset_train) is True


def test_check_data_train_fail():
    with pytest.raises(ValueError):
        check_data_train({"validation": "some_data"})


def test_check_data_structure_pass(valid_dataset):
    assert check_data_structure(valid_dataset) is True


def test_check_data_structure_fail():
    ds = Dataset.from_dict({"lyrics": ["abc"]})
    with pytest.raises(ValueError):
        check_data_structure(ds)


def test_check_data_types_pass(valid_dataset):
    assert check_data_types(valid_dataset) is True


def test_check_data_types_fail_lyrics_type():
    ds = Dataset.from_dict({"lyrics": [123], "genre": [["pop"]]})
    with pytest.raises(TypeError):
        check_data_types(ds)


def test_check_data_types_fail_empty_lyric():
    ds = Dataset.from_dict({"lyrics": [""], "genre": [["pop"]]})
    with pytest.raises(ValueError):
        check_data_types(ds)


def test_check_data_types_fail_genre_not_list():
    ds = Dataset.from_dict({"lyrics": ["song"], "genre": ["pop"]})
    with pytest.raises(TypeError):
        check_data_types(ds)


def test_check_data_types_fail_genre_empty_list():
    ds = Dataset.from_dict({"lyrics": ["song"], "genre": [[]]})
    with pytest.raises(ValueError):
        check_data_types(ds)


def test_check_data_types_fail_genre_element_type():
    ds = Dataset.from_dict({"lyrics": ["song"], "genre": [[123]]})
    with pytest.raises(TypeError):
        check_data_types(ds)


def test_encode_dataset(valid_dataset_train):
    encoded_data, num_labels = encode_dataset(valid_dataset_train)
    assert "labels" in encoded_data["train"].column_names
    assert num_labels == 2
    assert isinstance(encoded_data["train"]["labels"][0][0], (int, float))


def test_tokenize_batch(valid_dataset_train):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encoded_data, num_labels = encode_dataset(valid_dataset_train)
    batch_texts = {"lyrics": list(encoded_data["train"]["lyrics"])}
    batch = tokenize_batch(batch_texts, max_length=2, tokenizer=tokenizer)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert len(batch["input_ids"][0]) == 2


def test_convert_labels():
    batch = {"labels": [[1, 0], [0, 1]]}
    converted = convert_labels(batch)
    assert isinstance(converted["labels"][0][0], float)
