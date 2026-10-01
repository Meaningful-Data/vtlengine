"""
Macro Binding Tests

DuckDB expands a macro by copying each argument into every use of its parameter, so a
macro call nested in the argument of another one multiplies the SQL it binds at every
level (issue #1134). A call whose repeated argument would expand too far computes its
arguments once, in a list the macro is applied to. These tests pin when that happens
and check that the results and errors do not change.

Naming conventions:
- Identifiers: Id_1, Id_2, etc.
- Measures: Me_1, Me_2, etc.
"""

import time
import warnings
from typing import Dict, Tuple

import pandas as pd
import pytest

from vtlengine import run
from vtlengine.duckdb_transpiler import transpile
from vtlengine.duckdb_transpiler.Transpiler.macro_binding import bind_repeated_macro_arguments
from vtlengine.Exceptions import RunTimeError, SemanticError

# Patched by dotted path: the rewriter reads the threshold at each call
_THRESHOLD_PATH = "vtlengine.duckdb_transpiler.Transpiler.macro_binding.MIN_BOUND_ARGUMENT_SIZE"

# =============================================================================
# Helpers
# =============================================================================

N_MEASURES = 12

DATA_STRUCTURES = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
            ]
            + [
                {"name": f"Me_{i}", "type": "Number", "role": "Measure", "nullable": True}
                for i in range(1, N_MEASURES + 1)
            ],
        }
    ]
}


def _datapoints() -> Dict[str, pd.DataFrame]:
    """Positive values, nulls, -0.0 and values whose products overflow to inf - inf."""
    rows = {
        1: [1.5 + i for i in range(N_MEASURES)],
        2: [0.25 * (i + 1) for i in range(N_MEASURES)],
        3: [None if i % 3 == 0 else 7.0 / (i + 1) for i in range(N_MEASURES)],
        4: [-0.0 if i % 2 == 0 else 3.0 for i in range(N_MEASURES)],
        5: [1e200] * N_MEASURES,
        6: [123456.789 / (i + 1) for i in range(N_MEASURES)],
    }
    frame = pd.DataFrame.from_dict(
        rows, orient="index", columns=[f"Me_{i}" for i in range(1, N_MEASURES + 1)]
    )
    frame.insert(0, "Id_2", ["A", "B", "A", "B", "A", "B"])
    frame.insert(0, "Id_1", frame.index)
    return {"DS_1": frame.reset_index(drop=True)}


def _right_nested(op: str, n: int, start: int = 1) -> str:
    """``Me_1 op (Me_2 op (... op Me_n))``"""
    names = [f"Me_{i}" for i in range(start, start + n)]
    return "".join(f"{name} {op} (" for name in names[:-1]) + names[-1] + ")" * (n - 1)


def _nested(function: str, n: int, inner: str, *args: str) -> str:
    """``function(function(... function(inner, args) ..., args), args)``"""
    tail = "".join(f", {arg}" for arg in args)
    expr = inner
    for _ in range(n):
        expr = f"{function}({expr}{tail})"
    return expr


def _run(script: str, use_duckdb: bool) -> Dict[str, object]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    return run(
        script=script,
        data_structures=DATA_STRUCTURES,
        datapoints={k: v.copy() for k, v in _datapoints().items()},
        use_duckdb=use_duckdb,
    )


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(c for c in df.columns if c.startswith("Id"))
    return df.sort_values(ids).reset_index(drop=True) if ids else df


# Shapes whose nested SQL grows with every level, small enough to plan without binding
NESTED_SHAPES = {
    "right-nested sum": f"DS_r <- DS_1[calc Me_r := {_right_nested('+', 7)}];",
    "right-nested product and difference": (
        "DS_r <- DS_1[calc Me_r := Me_1 * (Me_2 - (Me_3 * (Me_4 - (Me_5 * Me_6))))];"
    ),
    "nested round": (
        "DS_r <- DS_1[calc Me_r := "
        + "round(" * 4
        + "Me_1 * Me_2 - Me_3 * Me_4"
        + "".join(f" + Me_{i}, 5)" for i in range(5, 9))
        + "];"
    ),
    "nested trunc": f"DS_r <- DS_1[calc Me_r := {_nested('trunc', 4, 'Me_1 / Me_2', '3')}];",
    "nested sqrt": f"DS_r <- DS_1[calc Me_r := {_nested('sqrt', 5, 'abs(Me_1)')}];",
    "nested ln": ("DS_r <- DS_1[calc Me_r := ln(ln(ln(abs(Me_1) + 20) + 5) + 5)];"),
    "nested log": "DS_r <- DS_1[calc Me_r := log(log(abs(Me_1) + 100, 2) + 10, 3)];",
    "nested power": "DS_r <- DS_1[calc Me_r := power(power(abs(Me_1) + 1, 0.5) + 1, 1.5)];",
    "nested division": (
        "DS_r <- DS_1[calc Me_r := Me_1 / (abs(Me_2) + 1 + Me_3 / (abs(Me_4) + 1 "
        "+ Me_5 / (abs(Me_6) + 1 + Me_7 / (abs(Me_8) + 1))))];"
    ),
    "nested mod": (
        "DS_r <- DS_1[calc Me_r := mod(Me_1, abs(Me_2) + 1 + mod(Me_3, abs(Me_4) + 1 "
        "+ mod(Me_5, abs(Me_6) + 1)))];"
    ),
    "tolerance equality of nested sums": (
        f"DS_r <- DS_1[calc Me_r := if ({_right_nested('+', 5)}) = "
        f"({_right_nested('+', 5, start=6)}) then 1.0 else 0.0];"
    ),
    "filter": f"DS_r <- DS_1[filter {_nested('sqrt', 5, 'abs(Me_1)')} > 1];",
    "aggr": f"DS_r <- DS_1[aggr Me_r := sum({_nested('sqrt', 4, 'abs(Me_1)')}) group by Id_2];",
    "analytic": (
        "DS_r <- DS_1[calc Me_r := "
        + _nested("sqrt", 4, "abs(sum(Me_1 over (partition by Id_2 order by Id_1)))")
        + "];"
    ),
    "dataset": "DS_r <- " + _nested("sqrt", 5, "abs(DS_1[keep Me_1])") + ";",
    "scalar": "sc_r <- " + _nested("sqrt", 5, "65536.0") + ";",
}


# =============================================================================
# Rewriting the SQL
# =============================================================================


class TestBindRepeatedMacroArguments:
    """How a macro call with a large repeated argument is rewritten."""

    def test_small_arguments_keep_the_sql(self) -> None:
        sql = 'SELECT vtl_round_sig(("Me_1" + "Me_2"), 15) AS "Me_r" FROM "DS_1"'

        assert bind_repeated_macro_arguments(sql) is sql

    def test_large_argument_is_bound_and_literals_stay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_THRESHOLD_PATH, 1)

        sql = bind_repeated_macro_arguments('SELECT vtl_round_sig(("Me_1" + "Me_2"), 15) FROM t')

        assert sql == (
            'SELECT list_transform([("Me_1" + "Me_2")], '
            "lambda __vtl_arg: vtl_round_sig(__vtl_arg, 15))[1] FROM t"
        )

    def test_every_other_argument_moves_into_the_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The body only reads the lambda parameter: an aggregate stays out of it."""
        monkeypatch.setattr(_THRESHOLD_PATH, 1)

        sql = bind_repeated_macro_arguments('SELECT vtl_div(("Me_1" * 2), sum("Me_2")) FROM t')

        assert sql == (
            "SELECT list_transform([{'a0': (\"Me_1\" * 2), 'a1': sum(\"Me_2\")}], "
            "lambda __vtl_arg: vtl_div(__vtl_arg.a0, __vtl_arg.a1))[1] FROM t"
        )

    def test_quoted_text_is_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_THRESHOLD_PATH, 1)

        sql = bind_repeated_macro_arguments("SELECT vtl_sqrt(length('a,)(b') + \"we(ird,\") FROM t")

        assert sql == (
            "SELECT list_transform([length('a,)(b') + \"we(ird,\"], "
            "lambda __vtl_arg: vtl_sqrt(__vtl_arg))[1] FROM t"
        )

    def test_unbalanced_sql_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_THRESHOLD_PATH, 1)
        sql = "SELECT vtl_sqrt((1 + 2) FROM t"

        assert bind_repeated_macro_arguments(sql) is sql

    def test_nested_calls_bind_once_they_grow(self) -> None:
        """The innermost calls are small and stay; the outer ones get bound."""
        sql = transpile(NESTED_SHAPES["nested sqrt"], DATA_STRUCTURES)[0][1]

        assert sql.count("list_transform(") == 1
        assert "lambda __vtl_arg: vtl_sqrt(__vtl_arg))[1]" in sql
        assert 'vtl_sqrt(vtl_sqrt(vtl_sqrt(vtl_sqrt(ABS("Me_1")))))' in sql

    @pytest.mark.parametrize(
        "script",
        [
            "DS_r <- DS_1[calc Me_r := Me_1 * Me_2 + Me_3];",
            "DS_r <- DS_1[calc Me_r := (Me_1 + Me_2) / Me_3 - Me_4];",
            "DS_r <- DS_1[calc Me_r := round(Me_1 / Me_2, 2) + sqrt(abs(Me_3))];",
            "DS_r <- DS_1[filter Me_1 + Me_2 = Me_3 * Me_4];",
        ],
    )
    def test_common_expressions_keep_the_sql(
        self, monkeypatch: pytest.MonkeyPatch, script: str
    ) -> None:
        bound = transpile(script, DATA_STRUCTURES)[0][1]
        monkeypatch.setattr(_THRESHOLD_PATH, 10**12)

        assert bound == transpile(script, DATA_STRUCTURES)[0][1]
        assert "list_transform" not in bound


# =============================================================================
# Results and errors
# =============================================================================


class TestBoundResults:
    """Bound calls give the results of the nested SQL and of the pandas engine."""

    @pytest.mark.parametrize("script", NESTED_SHAPES.values(), ids=NESTED_SHAPES.keys())
    def test_same_result_as_nested_sql_and_pandas(
        self, monkeypatch: pytest.MonkeyPatch, script: str
    ) -> None:
        monkeypatch.setattr(_THRESHOLD_PATH, 10**12)
        nested = _run(script, use_duckdb=True)
        # Every repeated argument is bound, however small
        monkeypatch.setattr(_THRESHOLD_PATH, 1)
        assert "list_transform" in transpile(script, DATA_STRUCTURES)[-1][1]
        bound = _run(script, use_duckdb=True)
        on_pandas = _run(script, use_duckdb=False)

        name = next(iter(nested))
        if name.startswith("sc_"):
            assert bound[name].value == nested[name].value == on_pandas[name].value
            return
        pd.testing.assert_frame_equal(
            _sorted(bound[name].data), _sorted(nested[name].data), check_exact=True
        )
        assert bound[name] == on_pandas[name]

    @pytest.mark.parametrize(
        "expression, code",
        [
            (_nested("sqrt", 3, "Me_1 - 100"), "2-1-15-2"),
            ("ln(ln(ln(Me_1 - Me_1)))", "2-1-15-8"),
            ("log(log(Me_1 + 100, 2), Me_1 - Me_1)", "2-1-15-3"),
            ("Me_1 / (Me_2 - Me_2 + (Me_3 - Me_3))", "2-1-15-6"),
        ],
    )
    @pytest.mark.parametrize("use_duckdb", [False, True])
    def test_domain_errors_are_raised(
        self, monkeypatch: pytest.MonkeyPatch, expression: str, code: str, use_duckdb: bool
    ) -> None:
        monkeypatch.setattr(_THRESHOLD_PATH, 1)
        script = f"DS_r <- DS_1[calc Me_r := {expression}];"
        assert "list_transform" in transpile(script, DATA_STRUCTURES)[0][1]

        with pytest.raises((SemanticError, RunTimeError)) as error:
            _run(script, use_duckdb)

        assert error.value.args[1] == code


# =============================================================================
# Planning
# =============================================================================


LARGE_SHAPES: Dict[str, Tuple[str, float]] = {
    "right-nested sum of 12": (f"DS_r <- DS_1[calc Me_r := {_right_nested('+', 12)}];", 10.0),
    "12 nested round": (
        "DS_r <- DS_1[calc Me_r := "
        + "round(" * 11
        + "Me_1"
        + "".join(f" + Me_{i}, 5)" for i in range(2, 13))
        + "];",
        10.0,
    ),
    "12 nested sqrt": (f"DS_r <- DS_1[calc Me_r := {_nested('sqrt', 12, 'Me_1')}];", 10.0),
    "12 nested ln": (f"DS_r <- DS_1[calc Me_r := {_nested('ln', 12, 'Me_1')}];", 10.0),
}


@pytest.mark.parametrize("script, seconds", LARGE_SHAPES.values(), ids=LARGE_SHAPES.keys())
def test_twelve_levels_plan_quickly(script: str, seconds: float) -> None:
    """Before the binding these took from seconds to minutes with no data."""
    start = time.perf_counter()
    run(script=script, data_structures=DATA_STRUCTURES, datapoints={}, use_duckdb=True)

    assert time.perf_counter() - start < seconds
