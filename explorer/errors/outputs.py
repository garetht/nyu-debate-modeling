from __future__ import annotations

from pathlib import Path

from explorer.cli.explorer_models import ConfigTypeLiteral

ConfigType = ConfigTypeLiteral

__all__ = [
    "InvalidGroupModeError",
    "OutputNotFoundError",
    "OutputStatsUnavailableError",
    "OutputsDirectoryMissingError",
]


class OutputsDirectoryMissingError(Exception):
    """Raised when the outputs directory does not exist."""

    def __init__(self, outputs_directory: Path) -> None:
        self.outputs_directory: Path = outputs_directory
        super().__init__(f"Outputs directory not found at {outputs_directory}")


class OutputNotFoundError(Exception):
    """Raised when the requested configuration output is absent."""

    def __init__(self, configuration: str) -> None:
        self.configuration: str = configuration
        super().__init__(f"Output configuration '{configuration}' was not found.")


class InvalidGroupModeError(Exception):
    """Raised when the provided group mode key is invalid."""

    def __init__(self, group_mode: str) -> None:
        self.group_mode: str = group_mode
        super().__init__(f"Unknown group_mode '{group_mode}'.")


class OutputStatsUnavailableError(Exception):
    """Raised when statistics are requested for a non-eval configuration."""

    def __init__(self, configuration: str, config_type: ConfigType) -> None:
        self.configuration: str = configuration
        self.config_type: ConfigType = config_type
        super().__init__(
            f"Statistics are only available for eval configurations. '{configuration}' is of type '{config_type}'."
        )
