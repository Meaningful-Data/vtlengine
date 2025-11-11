import json
from pathlib import Path
from typing import List

import pytest

from vtlengine.API import create_ast
from vtlengine.AST.DAG import DAGAnalyzer

override = False
data_path = Path(__file__).parent / "data"


def _discover_tests(data_root: Path) -> List[str]:
    return sorted(p.stem for p in (data_root / "vtl").iterdir() if p.is_file())


def _normalize_ds_structure(ds_structure):
    return json.loads(
        json.dumps(
            {
                "insertion": sorted(ds_structure["insertion"]),
                "deletion": sorted(ds_structure["deletion"]),
                "global_inputs": sorted(ds_structure["global_inputs"]),
                "persistent": sorted(ds_structure["persistent"]),
            }
        )
    )


tests = _discover_tests(data_path)


@pytest.mark.parametrize("test_code", tests)
def test_ds_structure(test_code):
    with open(data_path / "vtl" / f"{test_code}.vtl") as f:
        script = f.read()

    ds_structures = DAGAnalyzer.ds_structure(create_ast(script))

    if override:
        with open(data_path / "references" / f"{test_code}.json", "w") as f:
            json.dump(_normalize_ds_structure(ds_structures), f, indent=4)

    with open(data_path / "references" / f"{test_code}.json") as f:
        reference = json.load(f)

    normalized_ds_structures = _normalize_ds_structure(ds_structures)
    assert normalized_ds_structures == reference
