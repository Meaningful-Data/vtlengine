import json
from pathlib import Path
from typing import Any, Dict

import pytest

from vtlengine.API import create_ast
from vtlengine.AST.DAG import DAGAnalyzer

override = False
data_path = Path(__file__).parent / "data"


def _normalize_scheduling(schedule: Any) -> Dict[str, Any]:
    return json.loads(
        json.dumps(
            {
                "insertion": {k: sorted(v) for k, v in schedule.insertion.items()},
                "deletion": {k: sorted(v) for k, v in schedule.deletion.items()},
                "persistent": sorted(schedule.persistent),
            }
        )
    )


# Only keep tests with non-trivial scheduling (multiple insertion/deletion points).
NONTRIVIAL_TESTS = ["2", "3", "5", "6", "7", "8", "9", "10", "11", "13", "16", "35", "36"]


@pytest.mark.parametrize("test_code", NONTRIVIAL_TESTS)
def test_scheduling(test_code: str) -> None:
    with open(data_path / "vtl" / f"{test_code}.vtl") as f:
        script = f.read()

    schedule = DAGAnalyzer.ds_structure(create_ast(script))
    ref_path = data_path / "references" / "scheduling" / f"{test_code}.json"

    if override:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ref_path, "w") as f:
            json.dump(_normalize_scheduling(schedule), f, indent=4)

    with open(ref_path) as f:
        reference = json.load(f)

    assert _normalize_scheduling(schedule) == reference
