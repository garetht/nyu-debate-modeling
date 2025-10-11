import argparse
import dataclasses
from collections.abc import Mapping
from itertools import accumulate

import yaml

@dataclasses.dataclass
class Split:
    num_iters: int
    starting_index: int

@dataclasses.dataclass
class ConfigArgs:
    """Configuration arguments for the YAML validator."""
    configuration_name: str
    splits: int
    total_iterations: int
    filename: str
    output_directory: str = "run_orchestrator/runs/"
    ip_address: str = "0.0.0.0"

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def enum_representer(dumper, data):
    return dumper.represent_str(data.name.lower())

def generate_splits(args: ConfigArgs):
    splits = distribute_iterations(args.total_iterations, args.splits)
    return {
        "configurations": [
            {
                "instance_ip": args.ip_address,
                "home_dir": "/home/ubuntu/mars-arnesen-gareth",
                "configuration_filepath": "run_orchestrator/evals_generator/generated_configs/mars_experiments.yaml",
                "configurations":
                    {
                        "name": args.configuration_name,
                        "num_iters": s.num_iters,
                        "count": 1,
                        "starting_index": s.starting_index
                    }

            }
            for s in splits
        ]
    }


def write_configurations_to_disk(args: ConfigArgs, configurations: dict):
    write_config = {k: v for k, v in configurations.items()}

    with open(f"{args.output_directory}{args.filename}.yaml", 'w') as f:
        yaml.dump(write_config, f, default_flow_style=False, sort_keys=False, Dumper=NoAliasDumper)


def distribute_iterations(total_iterations: int, splits: int):
    remainder = total_iterations % splits
    iterations_per_split = total_iterations // splits

    iterations = [iterations_per_split] * splits
    for i in range(remainder):
        iterations[i] += 1

    return [
        Split(
            num_iters=iterations,
            starting_index=starting_index
        )
        for (iterations, starting_index)
        in zip(iterations, [0] + list(accumulate(iterations)))
    ]


def run_config_generator(args: ConfigArgs):
    configurations = generate_splits(args)
    write_configurations_to_disk(args, configurations)

def parse_args() -> ConfigArgs:
    """Parse command line arguments and return as a ConfigArgs dataclass."""
    parser = argparse.ArgumentParser(
        description="Validate a YAML configuration file against a JSON schema."
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Path to the YAML configuration file to generate."
    )
    parser.add_argument(
        "--configuration_name",
        type=str,
        help="Path to the YAML configuration file to generate."
    )
    parser.add_argument(
        "--total_iterations",
        type=int,
        help="Total iterations to go through"
    )
    parser.add_argument(
        "--splits",
        type=int,
        help="The number of splits to distribute the iterations to"
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        help="The IP address to use for each instance"
    )
    args = parser.parse_args()

    return ConfigArgs(filename=args.filename, configuration_name=args.configuration_name, total_iterations=args.total_iterations, splits=args.splits, ip_address=args.ip_address)


if __name__ == "__main__":
    config = parse_args()
    run_config_generator(config)
    print("Configuration generated!")
    print(f"python run_orchestrator/run_experiments.py start {config.filename}")
