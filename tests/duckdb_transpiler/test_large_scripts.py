"""
Large Script Tests

Scripts generated from validation rule sets chain many datasets in a single expression,
check each rule with its imbalance and repeat whole statements. These tests pin how the
DuckDB backend computes them and check that the results do not change: against the
pandas engine, and against the queries they were computed with before.

Naming conventions:
- Identifiers: Id_1, Id_2, etc.
- Measures: Me_1, Me_2, etc.
"""

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest

import vtlengine.duckdb_transpiler.Transpiler as transpiler_module
from vtlengine import run
from vtlengine.AST import Assignment, BinOp, Start, VarID
from vtlengine.DataTypes import Integer, Number, String
from vtlengine.duckdb_transpiler import transpile
from vtlengine.duckdb_transpiler.Transpiler import SQLTranspiler
from vtlengine.Model import Component, Dataset, Role

# =============================================================================
# Helpers
# =============================================================================


def _node(**kwargs: object) -> Dict[str, object]:
    return {"line_start": 1, "column_start": 1, "line_stop": 1, "column_stop": 10, **kwargs}


def _dataset(
    name: str,
    ids: Sequence[str] = ("Id_1",),
    measures: Sequence[str] = ("Me_1",),
    measure_type: type = Number,
    id_type: type = String,
    viral: Sequence[str] = (),
) -> Dataset:
    comps = {
        i: Component(name=i, data_type=id_type, role=Role.IDENTIFIER, nullable=False) for i in ids
    }
    for m in measures:
        comps[m] = Component(name=m, data_type=measure_type, role=Role.MEASURE, nullable=True)
    for v in viral:
        comps[v] = Component(name=v, data_type=String, role=Role.VIRAL_ATTRIBUTE, nullable=True)
    return Dataset(name=name, components=comps, data=None)


def _chain(names: List[str], ops: List[str]) -> BinOp:
    """Left-nested ``names[0] ops[0] names[1] ops[1] ...``."""
    expr = VarID(**_node(value=names[0]))
    for name, op in zip(names[1:], ops):
        expr = BinOp(**_node(left=expr, op=op, right=VarID(**_node(value=name))))
    return expr


def _transpile(datasets: Dict[str, Dataset], expr: BinOp, output: Dataset) -> str:
    transpiler = SQLTranspiler(
        input_datasets=datasets,
        output_datasets={"DS_r": output},
        input_scalars={},
        output_scalars={},
    )
    assignment = Assignment(**_node(left=VarID(**_node(value="DS_r")), op=":=", right=expr))
    return transpiler.transpile(Start(**_node(children=[assignment])))[0][1]


def _structure(name: str, measures: Sequence[str], measure_type: str) -> Dict[str, object]:
    components = [
        {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
        {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
    ]
    components += [
        {"name": m, "type": measure_type, "role": "Measure", "nullable": True} for m in measures
    ]
    return {"name": name, "DataStructure": components}


def _datapoints(
    measures: Sequence[str], measure_type: str, with_nulls: bool
) -> Dict[str, pd.DataFrame]:
    """Four operands over overlapping data points, some with null values."""
    rng = np.random.default_rng(7)
    frames = {}
    for k, dropped in enumerate(([], [3, 7], [11], [0, 39])):
        df = pd.DataFrame({"Id_1": np.arange(40) % 5, "Id_2": [f"K{i // 5}" for i in range(40)]})
        for m in measures:
            if measure_type == "Integer":
                df[m] = rng.integers(-1000, 1000, 40)
            else:
                df[m] = rng.normal(0, 1e6, 40) / 7.0
        if with_nulls and k == 0:
            df.loc[df.index % 4 == 1, measures[0]] = None
        frames[f"DS_{k + 1}"] = df.drop(index=dropped).reset_index(drop=True)
    return frames


def _run_both(
    script: str,
    data_structures: Dict[str, object],
    datapoints: Dict[str, pd.DataFrame],
    **kwargs: object,
) -> tuple:
    kwargs.setdefault("return_only_persistent", False)
    on_pandas = run(
        script=script,
        data_structures=data_structures,
        datapoints={k: v.copy() for k, v in datapoints.items()},
        **kwargs,
    )
    on_duckdb = run(
        script=script,
        data_structures=data_structures,
        datapoints={k: v.copy() for k, v in datapoints.items()},
        use_duckdb=True,
        **kwargs,
    )
    return on_pandas, on_duckdb


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(sorted(c for c in df.columns if c.startswith("Id"))).reset_index(
        drop=True
    )


# =============================================================================
# Chains of datasets
# =============================================================================


class TestLongChainTranspile:
    """Transpiling a long chain looks each structure up once."""

    def test_long_chain_resolves_each_structure_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each level of a chain resolves its operands again, which the memo absorbs."""
        names = [f"DS_{i}" for i in range(60)]
        datasets = {name: _dataset(name) for name in names}
        resolved = []
        original = SQLTranspiler._resolve_dataset_structure

        def counting(self: SQLTranspiler, node: object) -> object:
            resolved.append(node)
            return original(self, node)

        monkeypatch.setattr(SQLTranspiler, "_resolve_dataset_structure", counting)
        # ``*`` keeps the pairwise joins, where the structures are looked up level by level
        sql = _transpile(datasets, _chain(names, ["*"] * 59), datasets["DS_0"])

        assert sql.count(" JOIN ") == 59
        assert len(resolved) == len({id(node) for node in resolved})


class TestDatasetChainSQL:
    """A left-nested chain of dataset ``+``/``-`` is computed by one grouped query."""

    def test_chain_is_one_grouped_query(self) -> None:
        names = ["DS_1", "DS_2", "DS_3"]
        datasets = {name: _dataset(name) for name in names}

        sql = _transpile(datasets, _chain(names, ["-", "+"]), datasets["DS_1"])

        branches = " UNION ALL ".join(
            f'SELECT "Id_1", CAST("Me_1" AS DOUBLE) AS "Me_1", {k} AS "__vtl_chain_position__" '
            f'FROM "{name}" AS t'
            for k, name in enumerate(names)
        )
        assert sql == (
            'SELECT "Id_1", list_reduce(list("Me_1" ORDER BY "__vtl_chain_position__"), '
            "lambda __vtl_acc, __vtl_next, __vtl_pos: "
            "CASE WHEN [false, true, false][__vtl_pos] "
            "THEN vtl_round_sig((__vtl_acc - __vtl_next), 15) "
            'ELSE vtl_round_sig((__vtl_acc + __vtl_next), 15) END) AS "Me_1" '
            f'FROM ({branches}) AS t GROUP BY "Id_1" HAVING count(*) = 3'
        )

    def test_single_operator_chain_needs_no_operator_list(self) -> None:
        names = ["DS_1", "DS_2", "DS_3", "DS_4"]
        datasets = {name: _dataset(name, measure_type=Integer) for name in names}

        sql = _transpile(datasets, _chain(names, ["+", "+", "+"]), datasets["DS_1"])

        # Integer arithmetic stays exact: no rounding and no cast of the values
        assert "lambda __vtl_acc, __vtl_next, __vtl_pos: (__vtl_acc + __vtl_next))" in sql
        assert "CAST(" not in sql
        assert sql.endswith("HAVING count(*) = 4")

    def test_single_measure_takes_the_output_name(self) -> None:
        names = ["DS_1", "DS_2", "DS_3"]
        datasets = {name: _dataset(name) for name in names}

        sql = _transpile(datasets, _chain(names, ["+", "+"]), _dataset("DS_r", measures=["Me_9"]))

        assert sql.split(" FROM (")[0].endswith('AS "Me_9"')

    @pytest.mark.parametrize(
        "last_operand",
        [
            pytest.param(_dataset("DS_3", ids=("Id_1", "Id_2")), id="extra-identifier"),
            pytest.param(_dataset("DS_3", measure_type=Integer), id="other-measure-type"),
            pytest.param(_dataset("DS_3", measures=("Me_1", "Me_2")), id="extra-measure"),
            pytest.param(_dataset("DS_3", viral=("At_1",)), id="viral-attribute"),
        ],
    )
    def test_operands_that_differ_keep_the_joins(self, last_operand: Dataset) -> None:
        datasets = {"DS_1": _dataset("DS_1"), "DS_2": _dataset("DS_2"), "DS_3": last_operand}

        sql = _transpile(datasets, _chain(["DS_1", "DS_2", "DS_3"], ["+", "-"]), last_operand)

        assert "list_reduce" not in sql
        assert sql.count(" JOIN ") == 2

    def test_number_identifiers_keep_the_joins(self) -> None:
        names = ["DS_1", "DS_2", "DS_3"]
        datasets = {name: _dataset(name, id_type=Number) for name in names}

        sql = _transpile(datasets, _chain(names, ["+", "+"]), datasets["DS_1"])

        assert "list_reduce" not in sql

    def test_two_operands_keep_the_join(self) -> None:
        datasets = {name: _dataset(name) for name in ("DS_1", "DS_2")}

        sql = _transpile(datasets, _chain(["DS_1", "DS_2"], ["-"]), datasets["DS_1"])

        assert "list_reduce" not in sql
        assert " INNER JOIN " in sql


class TestDatasetChainResults:
    """A folded chain gives the pandas engine's result and the pairwise joins' result."""

    @pytest.mark.parametrize(
        "script, measures, measure_type",
        [
            ("DS_r := DS_1 - DS_2 + DS_3 - DS_4;", ["Me_1"], "Number"),
            ("DS_r := DS_1 + DS_2 + DS_3;", ["Me_1", "Me_2"], "Number"),
            ("DS_r := DS_1 - DS_2 - DS_3 - DS_4;", ["Me_1"], "Integer"),
            ("DS_r := DS_1 + DS_2 - DS_3 + DS_4 - DS_1;", ["Me_1"], "Integer"),
            ("DS_r := (DS_1 - DS_2) - (DS_3 + DS_4) + DS_2;", ["Me_1"], "Number"),
            ("DS_r := DS_1 - DS_2 + DS_3 - DS_4 * 2;", ["Me_1"], "Number"),
        ],
    )
    @pytest.mark.parametrize("with_nulls", [False, True])
    def test_same_result_as_pandas_and_joins(
        self,
        monkeypatch: pytest.MonkeyPatch,
        script: str,
        measures: List[str],
        measure_type: str,
        with_nulls: bool,
    ) -> None:
        data_structures = {
            "datasets": [_structure(f"DS_{k}", measures, measure_type) for k in range(1, 5)]
        }
        datapoints = _datapoints(measures, measure_type, with_nulls)

        on_pandas, folded = _run_both(script, data_structures, datapoints)
        monkeypatch.setattr(transpiler_module, "_MIN_FOLDED_CHAIN_OPERANDS", 10**9)
        joined = run(
            script=script,
            data_structures=data_structures,
            datapoints={k: v.copy() for k, v in datapoints.items()},
            use_duckdb=True,
            return_only_persistent=False,
        )

        pd.testing.assert_frame_equal(
            _sorted(folded["DS_r"].data), _sorted(joined["DS_r"].data), check_exact=True
        )
        assert folded["DS_r"] == on_pandas["DS_r"]


# =============================================================================
# Repeated statements
# =============================================================================


class TestRepeatedStatements:
    """A statement repeating an earlier statement's query copies that result."""

    @pytest.mark.parametrize("return_only_persistent", [False, True])
    def test_copy_outlives_the_first_result(self, return_only_persistent: bool) -> None:
        """DS_C repeats DS_A after DS_A's last reader, so DS_A has to be kept for it."""
        script = """
            DS_A := DS_1 - DS_2 + DS_3;
            DS_B <- DS_A * 2;
            DS_C := DS_1 - DS_2 + DS_3;
            DS_D <- DS_C + DS_B;
            DS_E <- DS_1 - DS_2 + DS_3;
        """
        data_structures = {
            "datasets": [_structure(f"DS_{k}", ["Me_1"], "Number") for k in range(1, 4)]
        }
        datapoints = _datapoints(["Me_1"], "Number", with_nulls=True)
        datapoints.pop("DS_4")

        on_pandas, on_duckdb = _run_both(
            script, data_structures, datapoints, return_only_persistent=return_only_persistent
        )

        assert set(on_duckdb) == set(on_pandas)
        for name, dataset in on_pandas.items():
            assert on_duckdb[name] == dataset

    def test_repeated_results_are_formatted_once(self) -> None:
        """DS_A is fetched, and formatted for output, only once DS_C has copied it."""
        script = """
            DS_A <- DS_1 - DS_2 + DS_3;
            DS_B <- DS_1 * 2;
            DS_C <- DS_1 - DS_2 + DS_3;
        """
        components = [
            {"name": "Id_1", "type": "Time_Period", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
        ]
        data_structures = {
            "datasets": [{"name": f"DS_{k}", "DataStructure": components} for k in range(1, 4)]
        }
        periods = ["2020A", "2021M03", "2021M11", "2022A"]
        datapoints = {
            f"DS_{k}": pd.DataFrame({"Id_1": periods, "Me_1": [1.5 * k, 2.0, None, 4.25 - k]})
            for k in range(1, 4)
        }

        for output_format in ("vtl", "sdmx_reporting", "sdmx_gregorian", "natural"):
            on_pandas, on_duckdb = _run_both(
                script,
                data_structures,
                datapoints,
                return_only_persistent=True,
                time_period_output_format=output_format,
            )
            for name in ("DS_A", "DS_C"):
                expected = _sorted(on_pandas[name].data)
                actual = _sorted(on_duckdb[name].data)
                assert actual["Id_1"].tolist() == expected["Id_1"].tolist(), output_format
                assert actual["Me_1"].astype("float64").tolist() == pytest.approx(
                    expected["Me_1"].astype("float64").tolist(), rel=1e-12, nan_ok=True
                )

    def test_repeated_results_saved_to_the_output_folder(self, tmp_path: Path) -> None:
        """DS_A is saved once DS_C has copied it, and both files hold the same rows."""
        script = """
            DS_A <- DS_1 - DS_2 + DS_3;
            DS_B <- DS_1 * 2;
            DS_C <- DS_1 - DS_2 + DS_3;
        """
        data_structures = {
            "datasets": [_structure(f"DS_{k}", ["Me_1"], "Number") for k in range(1, 4)]
        }
        datapoints = _datapoints(["Me_1"], "Number", with_nulls=True)
        datapoints.pop("DS_4")
        expected = run(
            script=script,
            data_structures=data_structures,
            datapoints={k: v.copy() for k, v in datapoints.items()},
        )

        run(
            script=script,
            data_structures=data_structures,
            datapoints={k: v.copy() for k, v in datapoints.items()},
            use_duckdb=True,
            output_folder=tmp_path,
        )

        saved = {name: _sorted(pd.read_csv(tmp_path / f"{name}.csv")) for name in ("DS_A", "DS_C")}
        pd.testing.assert_frame_equal(saved["DS_A"], saved["DS_C"])
        assert len(saved["DS_A"]) == len(expected["DS_A"].data)
        assert saved["DS_A"]["Me_1"].tolist() == pytest.approx(
            _sorted(expected["DS_A"].data)["Me_1"].astype("float64").tolist(),
            rel=1e-12,
            nan_ok=True,
        )


# =============================================================================
# Validations
# =============================================================================


class TestCheckSharedJoin:
    """A check whose imbalance combines the compared datasets reads them in one join."""

    @staticmethod
    def _sql(script: str) -> str:
        data_structures = {"datasets": [_structure(f"DS_{k}", ["Me_1"], "Number") for k in (1, 2)]}
        return transpile(script, data_structures)[0][1]

    def test_comparison_and_imbalance_share_the_join(self) -> None:
        sql = self._sql('DS_r := check(DS_1 >= DS_2 errorcode "E1" imbalance DS_1 - DS_2);')

        assert sql.count(" JOIN ") == 1
        assert 't."__vtl_imbalance__" AS "imbalance"' in sql

    @pytest.mark.parametrize("imbalance", ["DS_2 - DS_1", "DS_1 - DS_1", "DS_1 - DS_2 + DS_1"])
    def test_other_operands_keep_their_join(self, imbalance: str) -> None:
        sql = self._sql(f"DS_r := check(DS_1 >= DS_2 imbalance {imbalance});")

        assert "__vtl_imbalance__" not in sql

    @pytest.mark.parametrize("mode", ["invalid", "all"])
    @pytest.mark.parametrize("measure_type", ["Number", "Integer"])
    def test_same_result_as_pandas_and_separate_joins(
        self, monkeypatch: pytest.MonkeyPatch, mode: str, measure_type: str
    ) -> None:
        script = (
            f'DS_r := check(DS_1 >= DS_2 errorcode "E1" errorlevel 5 imbalance DS_1 - DS_2 {mode});'
        )
        data_structures = {
            "datasets": [_structure(f"DS_{k}", ["Me_1"], measure_type) for k in (1, 2)]
        }
        datapoints = _datapoints(["Me_1"], measure_type, with_nulls=True)
        datapoints = {name: datapoints[name] for name in ("DS_1", "DS_2")}

        on_pandas, shared = _run_both(script, data_structures, datapoints)
        monkeypatch.setattr(SQLTranspiler, "_check_shared_join", lambda self, node: None)
        separate = run(
            script=script,
            data_structures=data_structures,
            datapoints={k: v.copy() for k, v in datapoints.items()},
            use_duckdb=True,
            return_only_persistent=False,
        )

        pd.testing.assert_frame_equal(
            _sorted(shared["DS_r"].data), _sorted(separate["DS_r"].data), check_exact=True
        )
        assert shared["DS_r"] == on_pandas["DS_r"]
