from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI

from explorer.explorer_backend.server import app

DEFAULT_SCHEMA_OUTPUT_PATH = Path("schemas/explorer_web_server.schema.json")
DEFAULT_CLIENT_OUTPUT_DIR = Path("explorer/explorer-frontend/src/clients/explorer")
DEFAULT_GENERATOR_COMMAND: Sequence[str] = (
    "npx",
    "--yes",
    "openapi-typescript-codegen",
)
DEFAULT_GENERATOR_ARGS: Sequence[str] = ("--client", "fetch")


@dataclass(frozen=True)
class GenerationResult:
    schema_path: Path
    client_dir: Path
    generator_command: Sequence[str]
    generator_args: Sequence[str]


def export_openapi_schema(fastapi_app: FastAPI, output_path: Path) -> Dict[str, Any]:
    """
    Generate and persist the OpenAPI schema for the provided FastAPI application.

    Args:
        fastapi_app: The FastAPI application to introspect.
        output_path: Destination for the schema JSON file.

    Returns:
        The OpenAPI schema that was written to disk.
    """
    schema: Dict[str, Any] = fastapi_app.openapi()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return schema


def _ensure_generator_available(command: Sequence[str]) -> None:
    """Verify that the generator command is available before attempting to run it."""
    try:
        probe_command: List[str] = list(command) + ["--help"]
        subprocess.run(
            probe_command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        command_str = " ".join(command)
        raise RuntimeError(
            f"Unable to execute TypeScript client generator command: {command_str}"
        ) from exc


def generate_typescript_client(
    schema_path: Path,
    output_dir: Path,
    command: Sequence[str],
    extra_args: Sequence[str],
) -> Path:
    """
    Use an external OpenAPI code generator to produce a TypeScript client.

    Args:
        schema_path: Path to the OpenAPI schema file.
        output_dir: Destination directory for the generated TypeScript client.
        command: The command (e.g. ``npx --yes openapi-typescript-codegen``) to execute.
        extra_args: Additional CLI arguments passed to the generator.

    Returns:
        The directory where the client code was written.
    """
    _ensure_generator_available(command)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_command = list(command) + [
        "--useUnionTypes",
        "--input",
        str(schema_path),
        "--output",
        str(output_dir),
        *extra_args,
    ]
    try:
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as exc:
        command_str = " ".join(full_command)
        raise RuntimeError(f"TypeScript client generation failed: {command_str}") from exc
    return output_dir


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the OpenAPI schema and TypeScript client for "
            "explorer.explorer_backend.server."
        )
    )
    parser.add_argument(
        "--schema-output",
        type=Path,
        default=DEFAULT_SCHEMA_OUTPUT_PATH,
        help=f"Path for the OpenAPI schema JSON (default: {DEFAULT_SCHEMA_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--client-output-dir",
        type=Path,
        default=DEFAULT_CLIENT_OUTPUT_DIR,
        help=(
            "Directory for the generated TypeScript client "
            f"(default: {DEFAULT_CLIENT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--generator-command",
        type=str,
        nargs="+",
        default=list(DEFAULT_GENERATOR_COMMAND),
        help=(
            "Command used to invoke the TypeScript client generator. "
            "Defaults to 'npx --yes openapi-typescript-codegen'."
        ),
    )
    parser.add_argument(
        "--generator-args",
        type=str,
        nargs="*",
        default=list(DEFAULT_GENERATOR_ARGS),
        help="Extra arguments supplied to the generator command.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        dest="deprecated_output",
        help="Deprecated alias for --schema-output.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> GenerationResult:
    """
    CLI entry point that produces both the OpenAPI schema and a TypeScript client.

    Args:
        argv: Optional iterable of string arguments for easier testing.

    Returns:
        A GenerationResult describing where artifacts were written.
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    schema_output = Path(args.deprecated_output) if args.deprecated_output else Path(args.schema_output)
    client_output_dir = Path(args.client_output_dir)
    generator_command: Sequence[str] = tuple(args.generator_command)
    generator_args: Sequence[str] = tuple(args.generator_args)

    export_openapi_schema(app, schema_output)
    generate_typescript_client(
        schema_output,
        client_output_dir,
        generator_command,
        generator_args,
    )
    return GenerationResult(
        schema_path=schema_output,
        client_dir=client_output_dir,
        generator_command=generator_command,
        generator_args=generator_args,
    )


if __name__ == "__main__":
    try:
        result = main()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
    print(result.schema_path)
    print(result.client_dir)
