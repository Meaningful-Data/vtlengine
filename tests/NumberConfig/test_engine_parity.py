"""
Cross-engine parity tests for Number precision (issue #985).

Every test runs the same script and data through BOTH execution engines and
asserts they agree: output CSVs byte-for-byte and in-memory values exactly.
Number is float64 in both engines and every binary arithmetic result is rounded
to 15 significant digits by the shared kernel, so results must be identical,
not merely close.
"""

import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from vtlengine.API import run
from vtlengine.Exceptions import RunTimeError, SemanticError

NUMBER_DS: Dict[str, Any] = {
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

# 15-significant-digit values across magnitudes, the old DECIMAL(28,10) failure
# points (truncation below 1e-5, the 1e18 range ceiling, 1e30 overflow) and
# long division tails.
LONG_NUMBERS = [
    123456789012345.0,
    1.23456789012345,
    999999999999999.0,
    9.99999999999999e14,
    0.123456789012345,
    0.000123456789012345,
    1e-15,
    2.5e-300,
    1e15,
    9.9e17,
    1e18,
    1e30,
    -1.23456789012345,
    0.0,
    None,
    1 / 3,
    2 / 3,
    1.2345678901234567,  # 17 significant digits: float64 parse round-trip
]


def _number_df() -> pd.DataFrame:
    return pd.DataFrame({"Id_1": range(1, len(LONG_NUMBERS) + 1), "Me_1": LONG_NUMBERS})


def _run_both(
    script: str,
    data_structures: Dict[str, Any],
    df: pd.DataFrame,
    output_folder: Optional[Path] = None,
) -> Dict[bool, Dict[str, Any]]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = {}
    for use_duckdb in (False, True):
        folder = None
        if output_folder is not None:
            folder = output_folder / ("duckdb" if use_duckdb else "pandas")
            folder.mkdir(parents=True, exist_ok=True)
        results[use_duckdb] = run(
            script=script,
            data_structures=data_structures,
            datapoints={"DS_1": df.copy()},
            use_duckdb=use_duckdb,
            output_folder=folder,
            return_only_persistent=False,
        )
    return results


def _assert_csv_parity(output_folder: Path) -> None:
    """Output files must match byte-for-byte, ignoring row order (the DuckDB
    connection sets preserve_insertion_order=false)."""
    pandas_dir = output_folder / "pandas"
    duckdb_dir = output_folder / "duckdb"
    pandas_files = sorted(p.name for p in pandas_dir.iterdir())
    duckdb_files = sorted(p.name for p in duckdb_dir.iterdir())
    assert pandas_files == duckdb_files
    for name in pandas_files:
        pandas_lines = (pandas_dir / name).read_text().strip().splitlines()
        duckdb_lines = (duckdb_dir / name).read_text().strip().splitlines()
        assert pandas_lines[0] == duckdb_lines[0], f"{name}: header differs"
        assert sorted(pandas_lines[1:]) == sorted(duckdb_lines[1:]), f"{name}: rows differ"


def _assert_values_identical(results: Dict[bool, Dict[str, Any]], name: str) -> None:
    pandas_ds = results[False][name]
    duckdb_ds = results[True][name]
    id_cols = sorted(c.name for c in pandas_ds.components.values() if c.role.name == "IDENTIFIER")
    pandas_df = pandas_ds.data.sort_values(id_cols).reset_index(drop=True)
    duckdb_df = duckdb_ds.data.sort_values(id_cols).reset_index(drop=True)
    assert sorted(pandas_df.columns) == sorted(duckdb_df.columns)
    for col in pandas_df.columns:
        pandas_vals = pandas_df[col].tolist()
        duckdb_vals = duckdb_df[col].tolist()
        for i, (pv, dv) in enumerate(zip(pandas_vals, duckdb_vals)):
            if pd.isna(pv) or pd.isna(dv):
                both_null = bool(pd.isna(pv)) and bool(pd.isna(dv))
                assert both_null, f"{name}.{col}[{i}]: {pv!r} != {dv!r}"
            else:
                assert pv == dv, f"{name}.{col}[{i}]: {pv!r} != {dv!r}"


ARITHMETIC_SCRIPTS = {
    "copy": "DS_r <- DS_1;",
    "div_3": "DS_r <- DS_1[calc Me_1 := Me_1 / 3];",
    "div_7": "DS_r <- DS_1[calc Me_1 := Me_1 / 7];",
    "chain": "DS_r <- DS_1[calc Me_1 := ((Me_1 + 1.5) * 2.5 - 0.75) / 3];",
    "div_roundtrip": "DS_r <- DS_1[calc Me_1 := (Me_1 / 3) * 3];",
    "mult_long_literal": "DS_r <- DS_1[calc Me_1 := Me_1 * 1.000000000000001];",
    "literal_calc": "DS_r <- DS_1[calc Me_2 := 0.5];",
    "negative_zero": "DS_r <- DS_1[calc Me_1 := Me_1 * -1];",
    "mod_mixed_sign": "DS_r <- DS_1[calc Me_1 := mod(Me_1, 3)];",
    "power_int": "DS_r <- DS_1[calc Me_1 := power(Me_1, 2)];",
}


@pytest.mark.parametrize("script", ARITHMETIC_SCRIPTS.values(), ids=ARITHMETIC_SCRIPTS.keys())
def test_arithmetic_parity(script: str) -> None:
    with TemporaryDirectory() as tmp:
        results = _run_both(script, NUMBER_DS, _number_df(), output_folder=Path(tmp))
        _assert_csv_parity(Path(tmp))
    results = _run_both(script, NUMBER_DS, _number_df())
    _assert_values_identical(results, "DS_r")


def test_chained_rounding_matches_pandas_kernel() -> None:
    """(1/3)*3 must print 0.999999999999999 in BOTH engines: per-op rounding to
    15 significant digits is observable and must happen at the same points."""
    df = pd.DataFrame({"Id_1": [1], "Me_1": [1.0]})
    results = _run_both("DS_r <- DS_1[calc Me_1 := (Me_1 / 3) * 3];", NUMBER_DS, df)
    for use_duckdb in (False, True):
        assert results[use_duckdb]["DS_r"].data["Me_1"].tolist() == [0.999999999999999]


def test_unary_math_parity() -> None:
    """Sqrt is IEEE-correctly-rounded so raw float64 results are identical;
    exp/ln act as canaries for libm agreement on valid values."""
    values = [0.123456789012345, 1.5, 4.0, 12345.6789012345, 2.0]
    df = pd.DataFrame({"Id_1": range(1, len(values) + 1), "Me_1": values})
    script = """
        DS_sqrt <- DS_1[calc Me_1 := sqrt(Me_1)];
        DS_exp <- DS_1[calc Me_1 := exp(Me_1)];
        DS_ln <- DS_1[calc Me_1 := ln(Me_1)];
    """
    results = _run_both(script, NUMBER_DS, df)
    for name in ("DS_sqrt", "DS_exp", "DS_ln"):
        _assert_values_identical(results, name)


def test_aggregate_parity() -> None:
    values = [123456789012345.0, 1.23456789012345, 0.000123456789012345, 1 / 3, -7.5, None]
    ds = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Id_g", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    df = pd.DataFrame(
        {
            "Id_1": range(1, len(values) + 1),
            "Id_g": [1, 1, 1, 2, 2, 2],
            "Me_1": values,
        }
    )
    script = """
        DS_sum <- sum(DS_1 group by Id_g);
        DS_avg <- avg(DS_1 group by Id_g);
        DS_med <- median(DS_1 group by Id_g);
        DS_std <- stddev_pop(DS_1 group by Id_g);
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = {}
    for use_duckdb in (False, True):
        results[use_duckdb] = run(
            script=script,
            data_structures=ds,
            datapoints={"DS_1": df.copy()},
            use_duckdb=use_duckdb,
            return_only_persistent=False,
        )
    for name in ("DS_sum", "DS_avg", "DS_med", "DS_std"):
        _assert_values_identical(results, name)


def test_integer_arithmetic_stays_exact() -> None:
    """Integer + - * must stay exact beyond 2^53 in both engines (BIGINT in
    DuckDB, the int fast path in pandas). The input is 2^52 — exactly loadable
    on both engines — and the arithmetic carries the result past 2^53, where a
    float64 kernel would lose the +1."""
    ds = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Integer", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    df = pd.DataFrame({"Id_1": [1], "Me_1": [4503599627370496]})  # 2^52
    results = _run_both("DS_r <- DS_1[calc Me_1 := Me_1 * 4 + 1];", ds, df)
    for use_duckdb in (False, True):
        assert results[use_duckdb]["DS_r"].data["Me_1"].tolist() == [18014398509481985]


def test_scalar_csv_parity() -> None:
    script = """
        DS_r <- DS_1;
        sc_third <- 1/3;
        sc_big <- 123456789012345 + 0.5;
        sc_int <- 8 * 1.0;
    """
    df = pd.DataFrame({"Id_1": [1], "Me_1": [0.1]})
    with TemporaryDirectory() as tmp:
        results = _run_both(script, NUMBER_DS, df, output_folder=Path(tmp))
        pandas_scalars = (Path(tmp) / "pandas" / "_scalars.csv").read_text()
        duckdb_scalars = (Path(tmp) / "duckdb" / "_scalars.csv").read_text()
        assert pandas_scalars == duckdb_scalars
        assert "sc_third,0.333333333333333" in pandas_scalars
    for name in ("sc_third", "sc_big", "sc_int"):
        assert results[False][name].value == results[True][name].value


DOMAIN_ERROR_CASES = {
    "mod_zero": ("DS_r <- DS_1[calc Me_1 := mod(Me_1, 0)];", [5.0], "2-1-15-6"),
    "power_negative_base": ("DS_r <- DS_1[calc Me_1 := power(Me_1, 0.5)];", [-8.0], "2-1-15-2"),
    "power_zero_negative_exp": ("DS_r <- DS_1[calc Me_1 := power(Me_1, -1)];", [0.0], "2-1-15-6"),
    "sqrt_negative": ("DS_r <- DS_1[calc Me_1 := sqrt(Me_1)];", [-1.0], "2-1-15-2"),
    "ln_zero": ("DS_r <- DS_1[calc Me_1 := ln(Me_1)];", [0.0], "2-1-15-8"),
    "ln_negative": ("DS_r <- DS_1[calc Me_1 := ln(Me_1)];", [-1.0], "2-1-15-8"),
    "log_negative_value": ("DS_r <- DS_1[calc Me_1 := log(Me_1, 10)];", [-1.0], "2-1-15-8"),
    "log_negative_base": ("DS_r <- DS_1[calc Me_1 := log(Me_1, -2)];", [10.0], "2-1-15-3"),
}


@pytest.mark.parametrize(
    "script, values, error_code", DOMAIN_ERROR_CASES.values(), ids=DOMAIN_ERROR_CASES.keys()
)
def test_domain_errors_parity(script: str, values: list, error_code: str) -> None:  # type: ignore[type-arg]
    """Domain violations raise the same VTL error code in both engines (before
    #985 pandas crashed with raw ValueError/decimal.InvalidOperation and the
    Arrow fast path silently produced NaN)."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = pd.DataFrame({"Id_1": range(1, len(values) + 1), "Me_1": values})
    for use_duckdb in (False, True):
        with pytest.raises((SemanticError, RunTimeError)) as ctx:
            run(
                script=script,
                data_structures=NUMBER_DS,
                datapoints={"DS_1": df.copy()},
                use_duckdb=use_duckdb,
                return_only_persistent=False,
            )
        assert ctx.value.args[1] == error_code, (
            f"use_duckdb={use_duckdb}: expected {error_code}, got {ctx.value.args}"
        )


def test_valid_domain_edges_parity() -> None:
    """The guards must not fire on valid edge inputs."""
    df = pd.DataFrame({"Id_1": [1, 2], "Me_1": [-8.0, 0.0]})
    results = _run_both("DS_r <- DS_1[calc Me_1 := power(Me_1, 2)];", NUMBER_DS, df)
    for use_duckdb in (False, True):
        assert results[use_duckdb]["DS_r"].data["Me_1"].tolist() == [64.0, 0.0]
    df = pd.DataFrame({"Id_1": [1], "Me_1": [0.0]})
    results = _run_both("DS_r <- DS_1[calc Me_1 := sqrt(Me_1)];", NUMBER_DS, df)
    for use_duckdb in (False, True):
        assert results[use_duckdb]["DS_r"].data["Me_1"].tolist() == [0.0]
