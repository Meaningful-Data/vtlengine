from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DatasetSchedule:
    """Typed result of DAG dataset usage analysis.

    Tracks when datasets should be loaded/unloaded for memory-efficient execution.
    """

    insertion: Dict[int, List[str]] = field(default_factory=dict)
    deletion: Dict[int, List[str]] = field(default_factory=dict)
    global_inputs: List[str] = field(default_factory=list)
    persistent: List[str] = field(default_factory=list)
