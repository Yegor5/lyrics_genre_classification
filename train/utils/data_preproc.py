from datasets import Dataset, Features, Sequence, Value, load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer


features = Features({
    'input_ids': Sequence(Value('int64')),
    'attention_mask': Sequence(Value('int64')),
    'labels': Sequence(Value('float32'))
})

def encode_labels(example, mlb):
    example["labels"] = mlb.transform([example["genre"]])[0]
    return example

def read_dataset(data_path):
    data = load_dataset(data_path) 
    mlb = MultiLabelBinarizer()
    mlb.fit(data["train"]["genre"])
    data["train"] = data["train"].map(encode_labels, fn_kwargs={"mlb": mlb})
    data["train"] = data["train"].select_columns(["lyrics", "labels"])
    return data, len(mlb.classes_)

def tokenize_batch(batch, max_length, tokenizer):
    return tokenizer(batch["lyrics"], truncation=True, padding="max_length", max_length=max_length)

def convert_labels(batch):
    batch["labels"] = [list(map(float, l)) for l in batch["labels"]]
    return batch

def preproc_dataset(cfg):
    ds, num_labels = read_dataset(cfg["data_params"]["data_path"])
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["train_params"]["model_name"])
    tokenized_ds = ds.map(tokenize_batch, remove_columns=["lyrics"], batched=True, num_proc=2, fn_kwargs={"tokenizer": tokenizer, "max_length": cfg["train_params"]["max_length"]})
    tokenized_ds = tokenized_ds.map(convert_labels, features=features, batched=True, num_proc=2)

    split = tokenized_ds["train"].train_test_split(test_size=cfg["data_params"]["test_size"], seed=cfg["data_params"]["seed"])
    return split, num_labels