"""Thread-safety tests for DuckDB operations (issue #626).

Verifies that concurrent vtlengine executions using DuckDB work correctly
when called from multiple threads in the same process.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import pandas as pd

from vtlengine import run

DATA_STRUCTURES: Dict[str, Any] = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}

DATAPOINTS: Dict[str, Any] = {
    "DS_1": pd.DataFrame(
        {
            "Id_1": [1, 1, 2, 2, 3],
            "Id_2": ["A", "B", "A", "B", "A"],
            "Me_1": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
}


def _run_concurrent(
    script: str,
    workers: int = 2,
    external_routines: Optional[list] = None,  # type: ignore[type-arg]
) -> list:  # type: ignore[type-arg]
    def task(tid: int) -> Dict[str, Any]:
        return run(
            script=script,
            data_structures=DATA_STRUCTURES,
            datapoints=DATAPOINTS,
            external_routines=external_routines,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(task, i) for i in range(workers)]
        return [f.result() for f in as_completed(futures)]


class TestConcurrentAggregation:
    def test_concurrent_sum(self) -> None:
        results = _run_concurrent("DS_r <- sum(DS_1 group by Id_1);")
        assert len(results) == 2
        for r in results:
            df = r["DS_r"].data
            assert len(df) == 3
            sums = dict(zip(df["Id_1"], df["Me_1"]))
            assert sums == {1: 30.0, 2: 70.0, 3: 50.0}


class TestConcurrentAnalytic:
    def test_concurrent_first_value(self) -> None:
        script = "DS_r <- first_value(DS_1 over (partition by Id_1 order by Id_2 asc));"
        results = _run_concurrent(script)
        assert len(results) == 2
        for r in results:
            assert len(r["DS_r"].data) == 5


class TestConcurrentEval:
    def test_concurrent_eval(self) -> None:
        script = """
            DS_r <- eval(SQL1(DS_1) language "SQL"
                         returns dataset {
                           identifier<integer> Id_1,
                           identifier<string> Id_2,
                           measure<number> Me_1
                         });
        """
        routines = [{"name": "SQL1", "query": "SELECT Id_1, Id_2, Me_1 * 2 AS Me_1 FROM DS_1"}]
        results = _run_concurrent(script, external_routines=routines)
        assert len(results) == 2
        for r in results:
            df = r["DS_r"].data
            assert len(df) == 5
            doubled = dict(zip(df["Id_2"], df["Me_1"]))
            assert doubled["A"] in {20.0, 60.0, 100.0}
