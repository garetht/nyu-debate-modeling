from __future__ import annotations

from builtins import ProcessLookupError as BuiltinProcessLookupError

__all__ = [
    "ExplorerSSHStreamingError",
    "ExplorerSSHProcessError",
    "ExplorerSSHProcessNotFoundError",
    "ExplorerSSHProcessAmbiguousError",
]


class ExplorerSSHStreamingError(RuntimeError):
    """Raised when explorer SSH clients encounter an unrecoverable client-side streaming error."""


class ExplorerSSHProcessError(RuntimeError):
    """Base class for process discovery errors emitted by explorer SSH clients."""


class ExplorerSSHProcessNotFoundError(ExplorerSSHProcessError, BuiltinProcessLookupError):
    """Raised when explorer SSH clients cannot locate a matching remote process."""


class ExplorerSSHProcessAmbiguousError(ExplorerSSHProcessError):
    """Raised when explorer SSH clients find multiple remote processes for a lookup term."""
