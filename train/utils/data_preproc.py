import logging

from datasets import Features, Sequence, Value, load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


features = Features({
    'input_ids': Sequence(Value('int64')),
    'attention_mask': Sequence(Value('int64')),
    'labels': Sequence(Value('float32'))
})


def check_data_train(dataset):
    if 'train' not in dataset:
        raise ValueError("Отсутствует обязательный сплит train")
    return True


def check_data_structure(dataset):
    for column in ["lyrics", "genre"]:
        if column not in dataset.column_names:
            raise ValueError(f"Отсутствует обязательная колонка: {column}")
    return True


def check_data_types(dataset):
    for i, lyric in enumerate(dataset["lyrics"]):
        if not isinstance(lyric, str):
            raise TypeError(f"lyrics[{i}] должен быть строкой, получен {type(lyric)}")
        if len(lyric.strip()) == 0:
            raise ValueError(f"lyrics[{i}] не может быть пустой строкой")

    for i, genre in enumerate(dataset["genre"]):
        if not isinstance(genre, list):
            raise TypeError(f"genre[{i}] должен быть списком, получен {type(genre)}")
        if len(genre) == 0:
            raise ValueError(f"genre[{i}] не может быть пустым списком")
        for g in genre:
            if not isinstance(g, str):
                raise TypeError(f"Элемент genre[{i}] должен быть строкой, получен {type(g)}")

    return True


def read_dataset(data_path):
    try:
        data = load_dataset(data_path)
    except Exception as e:
        logger.error("Не удалось загрузить датасет: %s", e)
        raise
    check_data_train(data)
    check_data_structure(data["train"])
    check_data_types(data["train"])
    return data


def encode_labels(example, mlb):
    example["labels"] = mlb.transform([example["genre"]])[0]
    return example


def encode_dataset(data):
    mlb = MultiLabelBinarizer()
    mlb.fit(data["train"]["genre"])
    data["train"] = data["train"].map(encode_labels, fn_kwargs={"mlb": mlb})
    data["train"] = data["train"].select_columns(["lyrics", "labels"])
    return data, len(mlb.classes_)


def tokenize_batch(batch, max_length, tokenizer):
    return tokenizer(batch["lyrics"], truncation=True, padding="max_length", max_length=max_length)


def convert_labels(batch):
    batch["labels"] = [list(map(float, i)) for i in batch["labels"]]
    return batch


def preproc_dataset(cfg):
    ds = read_dataset(cfg["data_params"]["data_path"])
    ds, num_labels = encode_dataset(ds)

    tokenizer = AutoTokenizer.from_pretrained(cfg["train_params"]["model_name"])
    tokenized_ds = ds.map(tokenize_batch, remove_columns=["lyrics"], batched=True, num_proc=2, fn_kwargs={"tokenizer": tokenizer, "max_length": cfg["train_params"]["max_length"]})
    tokenized_ds = tokenized_ds.map(convert_labels, features=features, batched=True, num_proc=2)

    split = tokenized_ds["train"].train_test_split(test_size=cfg["data_params"]["test_size"], seed=cfg["data_params"]["seed"])
    return split, num_labels
