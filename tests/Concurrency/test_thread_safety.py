"""Thread-safety tests for DuckDB operations.

Verifies that concurrent vtlengine executions using DuckDB (aggregation, analytic,
and eval operations) work correctly when called from multiple threads.

See: https://github.com/Meaningful-Data/vtlengine/issues/626
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

import pandas as pd

from vtlengine import run

AGG_SCRIPT = """
    DS_r <- sum(DS_1 group by Id_1);
"""

AGG_DATA_STRUCTURES: Dict[str, Any] = {
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

AGG_DATAPOINTS = {
    "DS_1": pd.DataFrame(
        {
            "Id_1": [1, 1, 2, 2, 3],
            "Id_2": ["A", "B", "A", "B", "A"],
            "Me_1": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
}

ANALYTIC_SCRIPT = """
    DS_r <- first_value(DS_1 over (partition by Id_1 order by Id_2 asc));
"""

ANALYTIC_DATA_STRUCTURES: Dict[str, Any] = {
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

ANALYTIC_DATAPOINTS = {
    "DS_1": pd.DataFrame(
        {
            "Id_1": [1, 1, 2, 2, 3],
            "Id_2": ["A", "B", "A", "B", "A"],
            "Me_1": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
}

EVAL_SCRIPT = """
    DS_r <- eval(SQL1(DS_1) language "SQL"
                 returns dataset {
                   identifier<integer> Id_1,
                   measure<number> Me_1
                 });
"""

EVAL_DATA_STRUCTURES: Dict[str, Any] = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}

EVAL_DATAPOINTS = {"DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0]})}

EVAL_EXTERNAL_ROUTINES = [{"name": "SQL1", "query": "SELECT Id_1, Me_1 * 2 AS Me_1 FROM DS_1"}]


def _run_agg(thread_id: int) -> Dict[str, Any]:
    result = run(
        script=AGG_SCRIPT,
        data_structures=AGG_DATA_STRUCTURES,
        datapoints=AGG_DATAPOINTS,
    )
    return {"thread_id": thread_id, "result": result}


def _run_analytic(thread_id: int) -> Dict[str, Any]:
    result = run(
        script=ANALYTIC_SCRIPT,
        data_structures=ANALYTIC_DATA_STRUCTURES,
        datapoints=ANALYTIC_DATAPOINTS,
    )
    return {"thread_id": thread_id, "result": result}


def _run_eval(thread_id: int) -> Dict[str, Any]:
    result = run(
        script=EVAL_SCRIPT,
        data_structures=EVAL_DATA_STRUCTURES,
        datapoints=EVAL_DATAPOINTS,
        external_routines=EVAL_EXTERNAL_ROUTINES,
    )
    return {"thread_id": thread_id, "result": result}


class TestConcurrentAggregation:
    """Test concurrent aggregation operations (issue #626)."""

    def test_two_threads(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_run_agg, i) for i in range(2)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 2
        for r in results:
            ds_r = r["result"]["DS_r"].data
            assert len(ds_r) == 3
            assert set(ds_r["Id_1"].tolist()) == {1, 2, 3}
            expected_sums = {1: 30.0, 2: 70.0, 3: 50.0}
            for _, row in ds_r.iterrows():
                assert row["Me_1"] == expected_sums[row["Id_1"]]


class TestConcurrentAnalytic:
    """Test concurrent analytic operations (issue #626)."""

    def test_two_threads(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_run_analytic, i) for i in range(2)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 2
        for r in results:
            ds_r = r["result"]["DS_r"].data
            assert len(ds_r) == 5


class TestConcurrentMixed:
    """Test mixed concurrent operations (aggregation + analytic)."""

    def test_aggregation_and_analytic(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            agg_future = executor.submit(_run_agg, 0)
            analytic_future = executor.submit(_run_analytic, 1)
            agg_result = agg_future.result()
            analytic_result = analytic_future.result()

        assert len(agg_result["result"]["DS_r"].data) == 3
        assert len(analytic_result["result"]["DS_r"].data) == 5


class TestConcurrentEval:
    """Test concurrent external routine (eval) operations."""

    def test_two_threads(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_run_eval, i) for i in range(2)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 2
        for r in results:
            ds_r = r["result"]["DS_r"].data
            assert len(ds_r) == 3
            expected = {1: 20.0, 2: 40.0, 3: 60.0}
            for _, row in ds_r.iterrows():
                assert row["Me_1"] == expected[row["Id_1"]]


class TestConcurrentStress:
    """Stress test with higher concurrency."""

    def test_ten_concurrent_aggregations(self) -> None:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_run_agg, i) for i in range(10)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 10
        expected_sums = {1: 30.0, 2: 70.0, 3: 50.0}
        for r in results:
            ds_r = r["result"]["DS_r"].data
            assert len(ds_r) == 3
            for _, row in ds_r.iterrows():
                assert row["Me_1"] == expected_sums[row["Id_1"]]
