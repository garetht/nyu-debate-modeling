from dataclasses import dataclass
from typing import Optional


@dataclass
class ScriptConfig:
    local: bool = False
    num_iters: int = 1_000
    starting_index: int = 0
    log_level: str = "INFO"
    configuration_filepath: Optional[str] = None
    extant_debates_directory: Optional[str] = None
    configuration: str = ""
    test: bool = False
    load_only: bool = False
    suppress_graphs: bool = False
    local_rank: int = 0
    start_time: str = ""
    force_save_results: bool = False
    force_save_transcripts: bool = False
    transcripts_dir: Optional[str] = None
