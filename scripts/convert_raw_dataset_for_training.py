import argparse
import os
import pandas as pd
import yaml
import json

from script_utils import ScriptUtils, TrainType
ScriptUtils.setup_script()

from train.row_converter import RowConverter
from train.train_utils import TrainingConfig, TrainUtils
from data import SplitType


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a training dataset for a GPT-based judge model using YAML config"
    )
    parser.add_argument(
        "input_path", help="Path to the raw dataset file"
    )
    parser.add_argument(
        "output_jsonl", nargs='?', default="prepared_dataset.jsonl",
        help="Path where the prepared JSONL will be saved"
    )
    args = parser.parse_args()

    # Load YAML config from same directory as script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, 'config.yml')
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Initialize TrainingConfig from YAML
    config = TrainingConfig(**config_dict)

    # Override dataset file path with CLI argument
    # Assumes single DatasetConfig in config.dataset
    config.dataset[0].full_dataset_file_path = args.input_path

    # Instantiate RawDatasets using TrainUtils
    raw_datasets = TrainUtils.create_datasets(config=config)
    raw_dataset = raw_datasets[0]

    # Use the first speech structure defined in config
    speech_structure = config.speech_structure[0]

    records = []
    for row in raw_dataset.get_data(split=config.dataset[0].split_type):
        for model_inputs, output_text in RowConverter.convert_row(
            row=row,
            config=config,
            dataset=raw_dataset,
            speech_structure=speech_structure
        ):
            prompt = "\n".join(m.content for m in model_inputs).strip()
            completion = f" {output_text.strip()}\n"
            records.append({"prompt": prompt, "completion": completion})

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Prepared JSONL saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
