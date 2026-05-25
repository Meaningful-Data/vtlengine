from pathlib import Path

import duckdb
import pandas as pd

from vtlengine import run

SCRIPT = "DS_A <- DS_1 * 10;"

DATA_STRUCTURE = {
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


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    conn = duckdb.connect()
    try:
        conn.register("tmp_df", df)
        conn.execute(f"COPY (SELECT * FROM tmp_df) TO '{path}' (FORMAT PARQUET)")
    finally:
        conn.close()


def test_load_parquet_input_basic(tmp_path: Path) -> None:
    df = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0]})
    pq = tmp_path / "DS_1.parquet"
    _write_parquet(df, pq)

    result = run(
        script=SCRIPT,
        data_structures=DATA_STRUCTURE,
        datapoints={"DS_1": pq},
        use_duckdb=True,
    )

    assert "DS_A" in result
    out = result["DS_A"].data
    assert list(out["Id_1"]) == [1, 2, 3]
    assert list(out["Me_1"].astype(float)) == [100.0, 200.0, 300.0]


def test_parquet_path_is_not_treated_as_sdmx(tmp_path: Path) -> None:
    df = pd.DataFrame({"Id_1": [1], "Me_1": [42.0]})
    pq = tmp_path / "DS_1.parquet"
    _write_parquet(df, pq)

    result = run(
        script="DS_A <- DS_1;",
        data_structures=DATA_STRUCTURE,
        datapoints={"DS_1": str(pq)},  # string path, not Path object
        use_duckdb=True,
    )
    assert result["DS_A"].data["Me_1"].iloc[0] == 42.0
