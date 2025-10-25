from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Final, Mapping

from fastapi import FastAPI

from run_orchestrator.configuration_api import (
    DPOExperimentConfigurationModel,
    StandardExperimentConfigurationModel,
    app as configuration_app,
)

STANDARD_SCHEMA_PATH: Final[Path] = Path("run_orchestrator/experiment_orchestrator.schema.json")
DPO_SCHEMA_PATH: Final[Path] = Path("run_orchestrator/experiment_orchestrator_dpo.schema.json")
OPENAPI_SCHEMA_PATH: Final[Path] = Path("run_orchestrator/experiment_orchestrator.openapi.json")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Serialize the provided payload to the target path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_openapi_document(api_app: FastAPI) -> Dict[str, object]:
    """Leverage FastAPI to produce a full OpenAPI document for the configuration API."""
    openapi_document = api_app.openapi()
    if not isinstance(openapi_document, dict):
        msg = "FastAPI did not return a dictionary when generating OpenAPI."
        raise RuntimeError(msg)
    return openapi_document


def export_configuration_schemas() -> None:
    """Generate JSON schema files for both standard and DPO orchestrator configurations."""
    openapi_document = _build_openapi_document(configuration_app)
    _write_json(OPENAPI_SCHEMA_PATH, openapi_document)

    standard_schema = StandardExperimentConfigurationModel.model_json_schema(ref_template="#/$defs/{model}")
    standard_schema["$schema"] = "http://json-schema.org/draft-07/schema#"

    dpo_schema = DPOExperimentConfigurationModel.model_json_schema(ref_template="#/$defs/{model}")
    dpo_schema["$schema"] = "http://json-schema.org/draft-07/schema#"

    _write_json(STANDARD_SCHEMA_PATH, standard_schema)
    _write_json(DPO_SCHEMA_PATH, dpo_schema)


def main() -> None:
    """Generate configuration schemas using FastAPI and persist them to disk."""
    # Ensure models are referenced so FastAPI registers them in the OpenAPI document.
    _ = (StandardExperimentConfigurationModel, DPOExperimentConfigurationModel)
    export_configuration_schemas()


if __name__ == "__main__":
    main()
