"""The Components of every Data Set result are ordered as Identifiers, Measures, Viral
Attributes and other Attributes, by name within each group, on both engines (issue #1285).

``Dataset.__eq__`` sorts the data columns before comparing, so the order is checked here on
the names of the Components, the columns of the data and the header of the written files."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from tests.Helper import engine_params
from vtlengine import run
from vtlengine.DataTypes import Integer, Number, String
from vtlengine.Model import Component, Dataset, Role

# Declared out of order, so that neither the input order nor the alphabetical order of the
# names gives the expected one
DATA_STRUCTURES: Dict[str, Any] = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "At_1", "type": "String", "role": "Attribute", "nullable": True},
                {"name": "Me_2", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True},
                {"name": "At_2", "type": "String", "role": "Attribute", "nullable": True},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            ],
        }
    ]
}

DATAPOINTS = {
    "DS_1": pd.DataFrame(
        {
            "At_1": ["a", "b"],
            "Me_2": [1.5, 2.5],
            "Id_2": ["A", "B"],
            "VAt_1": ["v", "w"],
            "At_2": ["c", "d"],
            "Me_1": [10.0, 20.0],
            "Id_1": [1, 2],
        }
    )
}

SCRIPT = """
    DS_t := DS_1;
    DS_r <- DS_1;
    DS_c <- DS_1 [calc attribute At_0 := "x", Me_0 := Me_1 * 2];
"""

EXPECTED: Dict[str, List[str]] = {
    "DS_t": ["Id_1", "Id_2", "Me_1", "Me_2", "VAt_1", "At_1", "At_2"],
    "DS_r": ["Id_1", "Id_2", "Me_1", "Me_2", "VAt_1", "At_1", "At_2"],
    "DS_c": ["Id_1", "Id_2", "Me_0", "Me_1", "Me_2", "VAt_1", "At_0", "At_1", "At_2"],
}


def _run(use_duckdb: bool, **kwargs: Any) -> Dict[str, Any]:
    return run(
        script=SCRIPT,
        data_structures=DATA_STRUCTURES,
        datapoints=DATAPOINTS,
        use_duckdb=use_duckdb,
        **kwargs,
    )


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize("return_only_persistent", [True, False])
def test_in_memory_results_order(use_duckdb: bool, return_only_persistent: bool) -> None:
    result = _run(use_duckdb, return_only_persistent=return_only_persistent)
    expected_names = {"DS_r", "DS_c"} if return_only_persistent else set(EXPECTED)
    assert set(result) == expected_names
    for name in expected_names:
        assert list(result[name].components) == EXPECTED[name], name
        assert list(result[name].data.columns) == EXPECTED[name], name


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_output_folder_csv_header_order(use_duckdb: bool, tmp_path: Path) -> None:
    result = _run(use_duckdb, output_folder=tmp_path)
    for name in ("DS_r", "DS_c"):
        assert list(result[name].components) == EXPECTED[name], name
        header = (tmp_path / f"{name}.csv").read_text().splitlines()[0]
        assert header.split(",") == EXPECTED[name], name


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_output_folder_parquet_column_order(use_duckdb: bool, tmp_path: Path) -> None:
    if not use_duckdb:
        pytest.skip("Only the DuckDB engine writes Parquet")
    _run(use_duckdb, output_folder=tmp_path, output_format="parquet")
    for name in ("DS_r", "DS_c"):
        columns = list(pd.read_parquet(tmp_path / f"{name}.parquet").columns)
        assert columns == EXPECTED[name], name


@pytest.mark.parametrize("with_data", [True, False])
def test_sort_components(with_data: bool) -> None:
    components = {
        "At_1": Component("At_1", String, Role.ATTRIBUTE, True),
        "VAt_1": Component("VAt_1", String, Role.VIRAL_ATTRIBUTE, True),
        "Me_1": Component("Me_1", Number, Role.MEASURE, True),
        "Id_2": Component("Id_2", Integer, Role.IDENTIFIER, False),
        "Id_1": Component("Id_1", Integer, Role.IDENTIFIER, False),
    }
    data = pd.DataFrame({"At_1": ["a"], "VAt_1": ["v"], "Me_1": [1.0], "Id_2": [2], "Id_1": [1]})
    dataset = Dataset(name="DS_1", components=components, data=data if with_data else None)
    dataset.sort_components()
    expected = ["Id_1", "Id_2", "Me_1", "VAt_1", "At_1"]
    assert list(dataset.components) == expected
    if with_data:
        assert list(dataset.data.columns) == expected
        assert dataset.data.iloc[0].tolist() == [1, 2, 1.0, "v", "a"]
    else:
        assert dataset.data is None
