import yaml
from pathlib import Path


CONFIG_PATH = Path(__file__).parent.parent / "train" / "config.yaml"


def test_config_format():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    for section in ["train_params", "data_params", "save_params", "log_params"]:
        assert section in cfg, f"Отсутствует секция {section} в config.yaml"

    train_params = cfg["train_params"]
    assert "model_name" in train_params
    assert isinstance(train_params["model_name"], str)
    for key in ["max_length", "batch_size", "accum_size", "epochs"]:
        assert key in train_params and isinstance(train_params[key], int)
    for key in ["learning_rate", "weight_decay"]:
        assert key in train_params and isinstance(train_params[key], float)
    assert "seed" in train_params and isinstance(train_params["seed"], int)

    data_params = cfg["data_params"]
    assert "data_path" in data_params and isinstance(data_params["data_path"], str)
    assert "test_size" in data_params and isinstance(data_params["test_size"], float)
    assert "seed" in data_params and isinstance(data_params["seed"], int)

    save_params = cfg["save_params"]
    assert "save_path" in save_params and isinstance(save_params["save_path"], str)
    assert "hf_path" in save_params and isinstance(save_params["hf_path"], str)
    assert "save_hf" in save_params and isinstance(save_params["save_hf"], bool)

    log_params = cfg["log_params"]
    for key in ["log_steps", "test_size"]:
        assert key in log_params and isinstance(log_params[key], int)
    assert "log_path" in log_params and isinstance(log_params["log_path"], str)
