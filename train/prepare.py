import os

import argparse
from utils.data_preproc import preproc_dataset
import yaml


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    split, num_labels = preproc_dataset(cfg)
    processed_dir = os.path.join('data/processed', f'lyrics_genre_dataset_{cfg["data_params"]["data_size"]}')
    os.makedirs(processed_dir, exist_ok=True)
    split["train"].save_to_disk(os.path.join(processed_dir, "train"))
    split["test"].save_to_disk(os.path.join(processed_dir, "test"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)