import glob
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtantDebateIdentifiersExtractor:
    """Counter for debate identifiers from JSON files."""

    directory: str
    identifiers: list[str] = field(default_factory=list, init=False)

    def process_files(self) -> dict[str, int]:
        """
        Process all JSON files and count debate identifiers.

        Returns:
            dict: Dictionary with debate identifiers as keys and counts as values
        """
        json_files = glob.glob(str(Path(self.directory) / "*.json"))

        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self._extract_identifiers(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error processing {file_path}: {e}")
                continue

        return dict(Counter(self.identifiers))

    def _extract_identifiers(self, data: dict) -> None:
        """Extract debate identifiers from metadata."""
        metadata = data.get('metadata')

        if metadata is None:
            return

        # Handle metadata as array or single object
        if isinstance(metadata, list):
            for item in metadata:
                if isinstance(item, dict) and 'debate_identifier' in item:
                    self.identifiers.append(item['debate_identifier'])
        elif isinstance(metadata, dict) and 'debate_identifier' in metadata:
            self.identifiers.append(metadata['debate_identifier'])


def count_debate_identifiers(directory: str) -> dict[str, int]:
    """
    Count occurrences of debate_identifier from metadata in JSON files.

    Args:
        directory: Path to directory containing JSON files

    Returns:
        dict: Dictionary with debate identifiers as keys and counts as values
    """
    counter = ExtantDebateIdentifiersExtractor(directory)
    return counter.process_files()
