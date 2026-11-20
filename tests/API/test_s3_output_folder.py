"""An S3 URI as the output folder of run (GH #1361).

https://github.com/Meaningful-Data/vtlengine/issues/1361

The DuckDB engine must hand the URI to DuckDB as written, which writes it through httpfs,
and create no local folder. As a test cannot reach a bucket, the DuckDB connection is wrapped
so that the COPY statements to the URI are kept and write into a local folder instead.
"""

from contextlib import contextmanager

import duckdb
import pandas as pd
import pytest

import vtlengine.API as api
from tests.Helper import engine_params
from vtlengine import run
from vtlengine.Exceptions import DataLoadError, InputValidationException

BUCKET = "s3://some-bucket/results"

DATA_STRUCTURES = {
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


class _Bucket:
    """A DuckDB connection whose COPY statements to BUCKET write into a local folder."""

    def __init__(self, conn, folder, copies, error=None):
        self._conn, self._folder, self._error = conn, folder, error
        self.copies = copies

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def execute(self, statement, *args, **kwargs):
        if statement.startswith("COPY") and BUCKET in statement:
            self.copies.append(statement)
            if self._error is not None:
                raise self._error
            statement = statement.replace(BUCKET, self._folder.as_posix())
        return self._conn.execute(statement, *args, **kwargs)


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    """Runs in an empty working directory, with the bucket in a folder outside it."""
    folder = tmp_path / "bucket"
    folder.mkdir()
    working_directory = tmp_path / "cwd"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    real_connection = api.configured_connection
    wrapped = {"copies": []}

    @contextmanager
    def connection(*args, **kwargs):
        with real_connection(*args, **kwargs) as conn:
            yield _Bucket(conn, folder, wrapped["copies"], wrapped.get("error"))

    monkeypatch.setattr(api, "configured_connection", connection)
    return folder, working_directory, wrapped


def _run(script, output_folder, output_format, use_duckdb):
    return run(
        script=script,
        data_structures=DATA_STRUCTURES,
        datapoints={"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0]})},
        output_folder=output_folder,
        output_format=output_format,
        use_duckdb=use_duckdb,
    )


def _files(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


@pytest.mark.parametrize("output_format", ["csv", "parquet"])
@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1361_s3_output_folder(bucket, output_format, use_duckdb):
    """Cases 1 (csv) and 2 (parquet): DuckDB writes DS_r to the URI and nothing locally; the
    pandas engine rejects the URI and writes nothing."""
    folder, working_directory, wrapped = bucket
    if not use_duckdb:
        with pytest.raises(InputValidationException, match="0-1-1-2"):
            _run("DS_r <- DS_1;", BUCKET, output_format, use_duckdb)
        assert _files(working_directory) == []
        return

    result = _run("DS_r <- DS_1;", BUCKET, output_format, use_duckdb)

    assert sorted(result) == ["DS_r"]
    [copy] = wrapped["copies"]
    assert f" TO '{BUCKET}/DS_r.{output_format}' " in copy
    assert _files(working_directory) == []
    assert _files(folder) == [f"DS_r.{output_format}"]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1361_local_output_folder(bucket, use_duckdb):
    """Case 3: a local output folder is still written on both engines."""
    _, working_directory, _ = bucket

    _run("DS_r <- DS_1;", "results", "csv", use_duckdb)

    assert _files(working_directory) == ["results/DS_r.csv"]


def test_GH_1361_s3_output_folder_scalars(bucket, tmp_path):
    """The scalars of a run go to _scalars.csv in the URI too, as in a local folder."""
    folder, working_directory, wrapped = bucket
    script = 'DS_r <- DS_1; SC_1 <- 1 + 2; SC_2 <- "a,b"; SC_3 <- null;'

    _run(script, BUCKET, "csv", use_duckdb=True)
    _run(script, tmp_path / "local", "csv", use_duckdb=True)

    assert any(f" TO '{BUCKET}/_scalars.csv' " in copy for copy in wrapped["copies"])
    assert _files(working_directory) == []
    assert _files(folder) == ["DS_r.csv", "_scalars.csv"]
    written = pd.read_csv(folder / "_scalars.csv", dtype=str, keep_default_na=False)
    local = pd.read_csv(tmp_path / "local" / "_scalars.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(written, local)


def test_GH_1361_s3_write_error(bucket):
    """A write that fails, as without credentials, raises 0-3-1-19 naming the file."""
    _, working_directory, wrapped = bucket
    wrapped["error"] = duckdb.HTTPException("HTTP 403 (Forbidden)")

    with pytest.raises(DataLoadError, match="0-3-1-19") as error:
        _run("DS_r <- DS_1;", BUCKET, "csv", use_duckdb=True)

    assert f"Could not write {BUCKET}/DS_r.csv: HTTP 403 (Forbidden)" in error.value.args[0]
    assert _files(working_directory) == []
