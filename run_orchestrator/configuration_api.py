from __future__ import annotations

from ipaddress import IPv4Address
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


app = FastAPI(
    title="Experiment Orchestrator Configuration API",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


class BaseConfigurationGroupModel(BaseModel):
    """Fields shared between the standard and DPO configuration group definitions."""

    model_config = ConfigDict(extra="forbid")

    instance_ip: IPv4Address
    configuration_filepath: str = Field(default="experiments/configs/standard_experiment.yaml")
    extant_debates_directory: Optional[str] = None
    home_dir: Optional[str] = None


class StandardConfigurationDetailsModel(BaseModel):
    """Configuration details required for standard debate runs."""

    model_config = ConfigDict(extra="forbid")

    name: str
    num_iters: PositiveInt
    count: PositiveInt
    starting_index: Optional[int] = Field(default=None, ge=0)


class DPOConfigurationDetailsModel(BaseModel):
    """Configuration details where iteration parameters are optional for DPO runs."""

    model_config = ConfigDict(extra="forbid")

    name: str
    num_iters: PositiveInt = Field(default=1)
    count: PositiveInt = Field(default=1)
    starting_index: Optional[int] = Field(default=None, ge=0)


class StandardExperimentConfigurationGroupModel(BaseConfigurationGroupModel):
    """Experiment configuration group used when running the standard start command."""

    configurations: StandardConfigurationDetailsModel


class DPOExperimentConfigurationGroupModel(BaseConfigurationGroupModel):
    """Experiment configuration group used when running the start-dpo command."""

    configurations: DPOConfigurationDetailsModel


class StandardExperimentConfigurationModel(BaseModel):
    """Root model describing the structure of the standard experiment YAML file."""

    model_config = ConfigDict(
        title="Experiment Orchestrator Configuration Schema",
        extra="forbid",
    )

    configurations: List[StandardExperimentConfigurationGroupModel]


class DPOExperimentConfigurationModel(BaseModel):
    """Root model describing the structure of the start-dpo experiment YAML file."""

    model_config = ConfigDict(
        title="Experiment Orchestrator DPO Configuration Schema",
        extra="forbid",
    )

    configurations: List[DPOExperimentConfigurationGroupModel]


@app.post(
    "/configurations/start",
    response_model=StandardExperimentConfigurationModel,
    summary="Validate a standard experiment configuration",
)
def validate_standard_configuration(
    payload: StandardExperimentConfigurationModel,
) -> StandardExperimentConfigurationModel:
    """Echo the provided standard configuration so FastAPI can surface its schema."""
    return payload


@app.post(
    "/configurations/start-dpo",
    response_model=DPOExperimentConfigurationModel,
    summary="Validate a start-dpo experiment configuration",
)
def validate_dpo_configuration(
    payload: DPOExperimentConfigurationModel,
) -> DPOExperimentConfigurationModel:
    """Echo the provided DPO configuration so FastAPI can surface its schema."""
    return payload
