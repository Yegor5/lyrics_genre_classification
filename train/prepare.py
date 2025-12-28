from utils.data_preproc import preproc_dataset
import yaml
import argparse


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    split, num_labels = preproc_dataset(cfg)
    split.save_to_disk("data/processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)