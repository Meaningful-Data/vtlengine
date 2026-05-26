from pathlib import Path

import duckdb
import pandas as pd
import pytest

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


def test_save_datapoints_parquet(tmp_path: Path) -> None:
    """save_datapoints_duckdb writes a parquet file when output_format='parquet'."""
    from vtlengine.duckdb_transpiler.io._io import save_datapoints_duckdb

    conn = duckdb.connect()
    try:
        conn.execute("CREATE TABLE DS_OUT (Id_1 BIGINT, Me_1 DOUBLE)")
        conn.execute("INSERT INTO DS_OUT VALUES (1, 1.5), (2, 2.5)")

        save_datapoints_duckdb(
            conn,
            dataset_name="DS_OUT",
            output_path=tmp_path,
            delete_after_save=False,
            output_format="parquet",
        )

        produced = tmp_path / "DS_OUT.parquet"
        assert produced.exists()

        rows = conn.execute(
            f"SELECT Id_1, Me_1 FROM read_parquet('{produced}') ORDER BY Id_1"
        ).fetchall()
        assert rows == [(1, 1.5), (2, 2.5)]
    finally:
        conn.close()


def test_run_output_format_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0]})
    pq = tmp_path / "DS_1.parquet"
    _write_parquet(df, pq)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    run(
        script="DS_A <- DS_1 * 2;",
        data_structures=DATA_STRUCTURE,
        datapoints={"DS_1": pq},
        output_folder=out_dir,
        use_duckdb=True,
        output_format="parquet",
    )

    produced = out_dir / "DS_A.parquet"
    assert produced.exists(), list(out_dir.iterdir())

    conn = duckdb.connect()
    try:
        rows = conn.execute(
            f"SELECT Id_1, Me_1 FROM read_parquet('{produced}') ORDER BY Id_1"
        ).fetchall()
    finally:
        conn.close()

    assert rows == [(1, 20.0), (2, 40.0)]


def test_run_default_output_format_is_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0]})
    csv_path = tmp_path / "DS_1.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    run(
        script="DS_A <- DS_1 * 2;",
        data_structures=DATA_STRUCTURE,
        datapoints={"DS_1": csv_path},
        output_folder=out_dir,
        use_duckdb=True,
    )

    assert (out_dir / "DS_A.csv").exists()
    assert not (out_dir / "DS_A.parquet").exists()


TWO_DS_STRUCTURE = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        },
        {
            "name": "DS_2",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        },
    ]
}


def test_mixed_csv_and_parquet_inputs(tmp_path: Path) -> None:
    csv_df = pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0]})
    pq_df = pd.DataFrame({"Id_1": [1, 2], "Me_1": [1.0, 2.0]})
    csv_path = tmp_path / "DS_1.csv"
    pq_path = tmp_path / "DS_2.parquet"
    csv_df.to_csv(csv_path, index=False)
    _write_parquet(pq_df, pq_path)

    result = run(
        script="DS_A <- DS_1 + DS_2;",
        data_structures=TWO_DS_STRUCTURE,
        datapoints={"DS_1": csv_path, "DS_2": pq_path},
        use_duckdb=True,
    )

    out = result["DS_A"].data.sort_values("Id_1").reset_index(drop=True)
    assert list(out["Id_1"]) == [1, 2]
    assert list(out["Me_1"].astype(float)) == [11.0, 22.0]


def test_parquet_duplicate_identifier_raises(tmp_path: Path) -> None:
    from vtlengine.Exceptions import DataLoadError

    df = pd.DataFrame({"Id_1": [1, 1], "Me_1": [1.0, 2.0]})
    pq = tmp_path / "DS_1.parquet"
    _write_parquet(df, pq)

    with pytest.raises(DataLoadError) as excinfo:
        run(
            script="DS_A <- DS_1;",
            data_structures=DATA_STRUCTURE,
            datapoints={"DS_1": pq},
            use_duckdb=True,
        )
    assert "0-3-1-7" in str(excinfo.value)


def test_parquet_nonexistent_path_empty_table(tmp_path: Path) -> None:
    missing = tmp_path / "DS_1.parquet"  # never created
    result = run(
        script="DS_A <- DS_1;",
        data_structures=DATA_STRUCTURE,
        datapoints={"DS_1": missing},
        use_duckdb=True,
    )
    assert result["DS_A"].data is not None
    assert len(result["DS_A"].data) == 0


def test_parquet_output_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [1.0, 2.0, 3.0]})
    pq_in = tmp_path / "DS_1.parquet"
    _write_parquet(df, pq_in)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    run(
        script="DS_A <- DS_1 * 10;",
        data_structures=DATA_STRUCTURE,
        datapoints={"DS_1": pq_in},
        output_folder=out_dir,
        use_duckdb=True,
        output_format="parquet",
    )

    produced = out_dir / "DS_A.parquet"
    assert produced.exists()

    structure_for_dsa = {
        "datasets": [
            {
                "name": "DS_A",
                "DataStructure": DATA_STRUCTURE["datasets"][0]["DataStructure"],
            }
        ]
    }

    result = run(
        script="DS_B <- DS_A;",
        data_structures=structure_for_dsa,
        datapoints={"DS_A": produced},
        use_duckdb=True,
    )
    out = result["DS_B"].data.sort_values("Id_1").reset_index(drop=True)
    assert list(out["Me_1"].astype(float)) == [10.0, 20.0, 30.0]


def test_save_datapoints_invalid_output_format_raises(tmp_path: Path) -> None:
    """save_datapoints_duckdb raises InputValidationException for unsupported formats."""
    from vtlengine.duckdb_transpiler.io._io import save_datapoints_duckdb
    from vtlengine.Exceptions import InputValidationException

    conn = duckdb.connect()
    try:
        conn.execute("CREATE TABLE DS_OUT (Id_1 BIGINT)")
        with pytest.raises(InputValidationException) as excinfo:
            save_datapoints_duckdb(
                conn,
                dataset_name="DS_OUT",
                output_path=tmp_path,
                output_format="json",  # type: ignore[arg-type]
            )
        assert "0-1-1-16" in str(excinfo.value)
    finally:
        conn.close()


def test_run_parquet_without_duckdb_warns(tmp_path: Path) -> None:
    """output_format='parquet' with use_duckdb=False emits a UserWarning."""
    df = pd.DataFrame({"Id_1": [1], "Me_1": [10.0]})
    csv_path = tmp_path / "DS_1.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.warns(UserWarning, match="output_format='parquet' has no effect"):
        run(
            script="DS_A <- DS_1;",
            data_structures=DATA_STRUCTURE,
            datapoints={"DS_1": csv_path},
            output_folder=out_dir,
            use_duckdb=False,
            output_format="parquet",
        )
