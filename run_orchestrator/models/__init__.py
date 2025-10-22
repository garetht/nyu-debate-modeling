from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class ConfigurationDetails:
    name: str
    num_iters: int
    count: int
    starting_index: Optional[int] = None


@dataclass(frozen=True)
class ExperimentConfigurationGroup:
    instance_ip: str
    configurations: ConfigurationDetails
    configuration_filepath: str = "experiments/configs/standard_experiment.yaml"
    extant_debates_directory: Optional[str] = None
    home_dir: Optional[str] = None


@dataclass(frozen=True)
class ExperimentConfiguration:
    configurations: List[ExperimentConfigurationGroup]
