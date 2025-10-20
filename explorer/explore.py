#!/usr/bin/env python3
"""Entry point for the debate outputs explorer."""

from __future__ import annotations

from explorer.app import OutputsExplorerApp

__all__ = ["OutputsExplorerApp", "main"]


def main() -> None:
    """Launch the textual explorer application."""
    OutputsExplorerApp().run()


if __name__ == "__main__":
    main()
