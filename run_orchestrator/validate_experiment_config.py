
import argparse
import json
import yaml
from jsonschema import validate, ValidationError

def validate_config(schema_path, config_path):
    """
    Validates a YAML configuration file against a JSON schema.

    Args:
        schema_path (str): The path to the JSON schema file.
        config_path (str): The path to the YAML configuration file.
    """
    try:
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        with open(config_path, 'r') as config_file:
            config_data = yaml.safe_load(config_file)

        try:
            validate(instance=config_data, schema=schema)
            print(f"Validation successful in '{config_path}'")
        except ValidationError as e:
            print(f"Validation failed in '{config_path}':")
            print(e)

    except FileNotFoundError as e:
        print(f"Error: {e.strerror}: {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a YAML configuration file against a JSON schema.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/standard_experiment.yaml",
        help="Path to the YAML configuration file to validate."
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="schemas/standard_experiment.schema.json",
        help="Path to the JSON schema file."
    )
    args = parser.parse_args()

    validate_config(args.schema, args.config)
