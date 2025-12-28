import yaml
from pathlib import Path


config_path = Path(__file__).parent.parent / "params.yaml"


def test_config_format():
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    for section in ["train_params", "data_params", "save_params", "log_params", "eval_params"]:
        assert section in cfg, f"Отсутствует секция {section} в params.yaml"

    train_params = cfg["train_params"]
    assert "model_name" in train_params and isinstance(train_params["model_name"], str)
    for key in ["max_length", "batch_size", "accum_size", "seed"]:
        assert key in train_params and isinstance(train_params[key], int)
    for key in ["learning_rate", "weight_decay", "epochs"]:
        assert key in train_params and isinstance(train_params[key], float)

    data_params = cfg["data_params"]
    assert "data_size" in data_params and data_params["data_size"] in ["small", "medium", "large"]
    assert "test_size" in data_params and isinstance(data_params["test_size"], float)
    assert "seed" in data_params and isinstance(data_params["seed"], int)

    save_params = cfg["save_params"]
    for key in ["save_path", "hf_path"]:
        assert key in save_params and isinstance(save_params[key], str)
    assert "save_steps" in save_params and isinstance(save_params["save_steps"], int)
    assert "save_hf" in save_params and isinstance(save_params["save_hf"], bool)

    log_params = cfg["log_params"]
    for key in ["log_steps", "test_size"]:
        assert key in log_params and isinstance(log_params[key], int)
    assert "log_path" in log_params and isinstance(log_params["log_path"], str)
    
    eval_params = cfg["eval_params"]
    assert "eval_steps" in eval_params and isinstance(eval_params["eval_steps"], int)
    assert "metrics_path" in eval_params and isinstance(eval_params["metrics_path"], str)
