"""Thread-safety tests for DuckDB operations (issue #626).

Verifies that concurrent vtlengine executions using DuckDB work correctly
when called from multiple threads in the same process.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

from tests.Helper import _use_duckdb_backend
from vtlengine import run, semantic_analysis
from vtlengine.DataTypes import Number
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Scalar
from vtlengine.Operators.Numeric import AbsoluteValue
from vtlengine.Utils import _duckdb
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.Utils._duckdb import query_connection
from vtlengine.ViralPropagation import (
    ViralPropagationRegistry,
    get_current_registry,
    set_current_registry,
)

# run() executes these scripts on the pandas engine, so the pandas pass is enough
pytestmark = pytest.mark.skipif(
    _use_duckdb_backend(), reason="Runs the pandas engine, in the pandas pass only"
)

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


class TestConcurrentParsing:
    """The C++ parser keeps the last parse tree in process-global state and hands
    Python raw pointers into it; a concurrent parse must not free it mid-traversal.
    Different scripts per thread surface any cross-contamination, and a tiny GIL
    switch interval widens the race window so a regression fails reliably here."""

    def test_concurrent_mixed_scripts(self) -> None:
        expectations = {
            "DS_r <- sum(DS_1 group by Id_1);": {1: 30.0, 2: 70.0, 3: 50.0},
            "DS_r <- max(DS_1 group by Id_1);": {1: 20.0, 2: 40.0, 3: 50.0},
            "DS_r <- min(DS_1 group by Id_1);": {1: 10.0, 2: 30.0, 3: 50.0},
        }
        scripts = list(expectations)

        def task(idx: int) -> Dict[Any, Any]:
            script = scripts[idx % len(scripts)]
            df = run(script=script, data_structures=DATA_STRUCTURES, datapoints=DATAPOINTS)[
                "DS_r"
            ].data
            assert dict(zip(df["Id_1"], df["Me_1"])) == expectations[script]
            return {}

        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            for _ in range(40):
                with ThreadPoolExecutor(max_workers=6) as executor:
                    futures = [executor.submit(task, i) for i in range(6)]
                    for future in as_completed(futures):
                        future.result()
        finally:
            sys.setswitchinterval(previous_interval)


class TestConcurrentAnalysis:
    """run() analyses the structures of the whole script before it computes the data, and the
    operators of one thread must compute the data while another thread analyses its script
    (issue #1352)."""

    def test_concurrent_runs_compute_data(self) -> None:
        script = "\n".join(f"DS_r{i} <- DS_1 [ calc Me_{i} := Me_1 + {i} ];" for i in range(1, 21))

        def task(_: int) -> None:
            results = run(script=script, data_structures=DATA_STRUCTURES, datapoints=DATAPOINTS)
            assert len(results) == 20
            for result in results.values():
                assert result.data is not None
                assert len(result.data) == 5

        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            for _ in range(10):
                with ThreadPoolExecutor(max_workers=6) as executor:
                    futures = [executor.submit(task, i) for i in range(6)]
                    for future in as_completed(futures):
                        future.result()
        finally:
            sys.setswitchinterval(previous_interval)

    def test_failed_run_leaves_computation_on(self) -> None:
        """A run that raises in its analysis gives back to its thread the mode it had: the
        operators that the thread calls after it compute their data (issue #1352)."""
        with pytest.raises(SemanticError, match="1-1-1-10"):
            run(
                script="DS_r <- DS_1 [ calc Me_2 := Me_9 ];",
                data_structures=DATA_STRUCTURES,
                datapoints=DATAPOINTS,
            )
        assert AbsoluteValue.analyze(Scalar(name="SC_1", data_type=Number, value=-2.0)).value == 2.0
        results = run(
            script="DS_r <- DS_1 + 1;", data_structures=DATA_STRUCTURES, datapoints=DATAPOINTS
        )
        assert list(results["DS_r"].data["Me_1"]) == [11.0, 21.0, 31.0, 41.0, 51.0]

    def test_semantic_analysis_inside_run(self) -> None:
        """A semantic analysis called while a run computes a statement (from an operator, as an
        external routine or a callback could) does not stop the computation of the statements
        that follow (issue #1352)."""
        evaluate = AbsoluteValue.evaluate

        def evaluate_with_analysis(cls: Any, *args: Any, **kwargs: Any) -> Any:
            semantic_analysis(script="DS_r <- DS_1 * 2;", data_structures=DATA_STRUCTURES)
            return evaluate(*args, **kwargs)

        script = "DS_r1 <- abs(DS_1);\nDS_r2 <- DS_1 + 1;"
        with patch.object(AbsoluteValue, "evaluate", classmethod(evaluate_with_analysis)):
            results = run(script=script, data_structures=DATA_STRUCTURES, datapoints=DATAPOINTS)
        assert list(results["DS_r1"].data["Me_1"]) == [10.0, 20.0, 30.0, 40.0, 50.0]
        assert results["DS_r2"].data is not None
        assert list(results["DS_r2"].data["Me_1"]) == [11.0, 21.0, 31.0, 41.0, 51.0]


class TestConcurrentJoins:
    """The join operators keep their reference Data Set and the operators number their virtual
    Data Sets and Components for the thread that runs them, so valid join scripts run in
    concurrent threads give their results (issue #1352)."""

    def test_concurrent_joins(self) -> None:
        structures = {
            "datasets": [
                *DATA_STRUCTURES["datasets"],
                {
                    "name": "DS_2",
                    "DataStructure": [
                        {
                            "name": "Id_1",
                            "type": "Integer",
                            "role": "Identifier",
                            "nullable": False,
                        },
                        {"name": "Me_2", "type": "Number", "role": "Measure", "nullable": True},
                    ],
                },
            ]
        }
        datapoints = {
            **DATAPOINTS,
            "DS_2": pd.DataFrame({"Id_1": [1, 2, 3], "Me_2": [1.0, 2.0, 3.0]}),
        }
        script = "\n".join(
            f"DS_r{i} <- inner_join(DS_1, DS_2 calc Me_{i + 3} := Me_1 + Me_2);" for i in range(5)
        )

        def task(_: int) -> None:
            results = run(script=script, data_structures=structures, datapoints=datapoints)
            for i in range(5):
                data = results[f"DS_r{i}"].data
                assert list(data[f"Me_{i + 3}"]) == [11.0, 21.0, 32.0, 42.0, 53.0]

        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(task, i) for i in range(24)]
                for future in as_completed(futures):
                    future.result()
        finally:
            sys.setswitchinterval(previous_interval)

    def test_virtual_names_and_rules_per_thread(self) -> None:
        VirtualCounter.reset()
        assert VirtualCounter._new_ds_name() == "__VDS_1__"
        registry = ViralPropagationRegistry()
        set_current_registry(registry)

        def other_thread() -> tuple:  # type: ignore[type-arg]
            return (
                VirtualCounter._new_ds_name(),
                VirtualCounter._new_dc_name(),
                get_current_registry() is registry,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(other_thread).result() == ("__VDS_1__", "__VDC_1__", False)
        assert VirtualCounter._new_ds_name() == "__VDS_2__"
        assert VirtualCounter.component_count == 0
        assert get_current_registry() is registry
        VirtualCounter.reset()


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


class TestQueryConnection:
    """The pandas operators query their frames on connections to one shared database."""

    def test_registered_frames_stay_private(self) -> None:
        with query_connection() as first, query_connection() as second:
            first.register("df", pd.DataFrame({"x": [1, 2]}))
            assert first.execute("SELECT sum(x) FROM df").fetchone() == (3,)
            with pytest.raises(duckdb.CatalogException):
                second.execute("SELECT * FROM df")
        with query_connection() as third, pytest.raises(duckdb.CatalogException):
            third.execute("SELECT * FROM df")

    def test_fatal_error_opens_another_database(self) -> None:
        with query_connection():
            before = _duckdb._database.connection
        with pytest.raises(duckdb.FatalException):
            _fail_in_a_query()
        with query_connection():
            after = _duckdb._database.connection
        assert before is not None
        assert after is not None
        assert after is not before


def _fail_in_a_query() -> None:
    with query_connection():
        raise duckdb.FatalException("invalidated")
