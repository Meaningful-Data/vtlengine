"""Cross-backend output dtype parity (issue #976).

DuckDB results must materialize the declared component dtypes exactly like
the pandas backend does via ``Dataset.enforce_dtypes()``.
"""

from pathlib import Path
from typing import Any, Dict

import duckdb
import pandas as pd

from vtlengine import run

DPR_SCRIPT = """
    define datapoint ruleset DR_1 (variable Me_1) is
        R_1: Me_1 >= 0 errorcode "R_1" errorlevel 2
    end datapoint ruleset;

    DS_r <- check_datapoint(DS_1, DR_1);
"""

DPR_STRUCTURES: Dict[str, Any] = {
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


def _dpr_datapoints() -> Dict[str, pd.DataFrame]:
    return {"DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10, -5, -30]})}


PASSTHROUGH_SCRIPT = "DS_r <- DS_1;"

PASSTHROUGH_STRUCTURES: Dict[str, Any] = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "Me_2", "type": "Integer", "role": "Measure", "nullable": True},
                {"name": "Me_3", "type": "String", "role": "Measure", "nullable": True},
                {"name": "Me_4", "type": "Boolean", "role": "Measure", "nullable": True},
                {"name": "Me_5", "type": "Date", "role": "Measure", "nullable": True},
            ],
        }
    ]
}


def _passthrough_datapoints() -> Dict[str, pd.DataFrame]:
    return {
        "DS_1": pd.DataFrame(
            {
                "Id_1": [1, 2],
                "Me_1": [19.5, None],
                "Me_2": [2, None],
                "Me_3": ["A", None],
                "Me_4": [True, None],
                "Me_5": ["2020-01-01", None],
            }
        )
    }


def test_check_datapoint_errorlevel_matches_declared_number() -> None:
    result = run(
        script=DPR_SCRIPT,
        data_structures=DPR_STRUCTURES,
        datapoints=_dpr_datapoints(),
        use_duckdb=True,
    )
    errorlevel = result["DS_r"].data["errorlevel"]
    assert str(errorlevel.dtype) == "double[pyarrow]"


def test_output_dtypes_identical_across_backends() -> None:
    dtypes = {}
    for use_duckdb in (False, True):
        result = run(
            script=PASSTHROUGH_SCRIPT,
            data_structures=PASSTHROUGH_STRUCTURES,
            datapoints=_passthrough_datapoints(),
            use_duckdb=use_duckdb,
        )
        dtypes[use_duckdb] = {c: str(t) for c, t in result["DS_r"].data.dtypes.items()}
    assert dtypes[True] == dtypes[False]


CSV_SCRIPT = """
    define datapoint ruleset DR_1 (variable Me_1) is
        R_1: Me_1 >= 0 errorcode "R_1" errorlevel 2
    end datapoint ruleset;

    DS_r <- check_datapoint(DS_1, DR_1 all_measures);
"""


def _csv_datapoints() -> Dict[str, pd.DataFrame]:
    return {"DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10.5, -5, None]})}


def test_csv_output_identical_across_backends(tmp_path: Path) -> None:
    contents = {}
    for use_duckdb in (False, True):
        folder = tmp_path / ("duckdb" if use_duckdb else "pandas")
        folder.mkdir()
        run(
            script=CSV_SCRIPT,
            data_structures=DPR_STRUCTURES,
            datapoints=_csv_datapoints(),
            use_duckdb=use_duckdb,
            output_folder=folder,
        )
        contents[use_duckdb] = sorted((folder / "DS_r.csv").read_text().splitlines())
    assert contents[True] == contents[False]


def test_parquet_output_types_match_declared(tmp_path: Path) -> None:
    run(
        script=DPR_SCRIPT,
        data_structures=DPR_STRUCTURES,
        datapoints=_dpr_datapoints(),
        use_duckdb=True,
        output_folder=tmp_path,
        output_format="parquet",
    )
    conn = duckdb.connect()
    rows = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{tmp_path / 'DS_r.parquet'}')"
    ).fetchall()
    types = {name: sql_type for name, sql_type, *_ in rows}
    assert types["errorlevel"] == "DOUBLE"
