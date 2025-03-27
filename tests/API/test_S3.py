import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from vtlengine.Model.dataframe_resolver import DataFrame, Series, isnull
import pandas as pd
import pytest

from vtlengine import DataTypes, run
from vtlengine.Exceptions import InputValidationException
from vtlengine.files.output import TimePeriodRepresentation, save_datapoints
from vtlengine.files.parser import load_datapoints
from vtlengine.Model import Component, Dataset, Role

base_path = Path(__file__).parent
filepath_output = base_path / "data" / "DataSet" / "output"
filepath_datastructure = base_path / "data" / "DataStructure" / "input"

params = [
    (
        Dataset(
            name="test_dataset",
            components={
                "Id_1": Component(
                    name="Id_1",
                    data_type=DataTypes.Integer,
                    role=Role.IDENTIFIER,
                    nullable=False,
                ),
                "Id_2": Component(
                    name="Id_2",
                    data_type=DataTypes.String,
                    role=Role.IDENTIFIER,
                    nullable=False,
                ),
            },
            data=DataFrame(columns=["Id_1", "Id_2"]),
        ),
        filepath_output / "test_dataset.csv",
    ),
]


@patch("pandas.DataFrame.to_csv")
def test_save_datapoints_without_data_mock(mock_csv):
    dataset = Dataset(
        name="test_dataset",
        components={
            "Id_1": Component(
                name="Id_1",
                data_type=DataTypes.Integer,
                role=Role.IDENTIFIER,
                nullable=False,
            ),
            "Id_2": Component(
                name="Id_2",
                data_type=DataTypes.String,
                role=Role.IDENTIFIER,
                nullable=False,
            ),
        },
        data=None,
    )
    output_path = "s3://path/to/output"

    save_datapoints(None, dataset, output_path)

    expected_path = "s3://path/to/output/test_dataset.csv"
    mock_csv.assert_called_once_with(expected_path, index=False)


@patch("pandas.DataFrame.to_csv")
def test_save_datapoints_with_data_mock(mock_csv):
    mock_data = DataFrame(columns=["Id_1", "Id_2"])
    dataset = Dataset(
        name="test_dataset",
        components={
            "Id_1": Component(
                name="Id_1",
                data_type=DataTypes.Integer,
                role=Role.IDENTIFIER,
                nullable=False,
            ),
            "Id_2": Component(
                name="Id_2",
                data_type=DataTypes.String,
                role=Role.IDENTIFIER,
                nullable=False,
            ),
        },
        data=mock_data,
    )
    output_path = "s3://path/to/output/"

    save_datapoints(None, dataset, output_path)

    expected_path = "s3://path/to/output/test_dataset.csv"
    mock_csv.assert_called_once_with(expected_path, index=False)


@patch("pandas.DataFrame.to_csv")
def test_save_datapoints_with_data_and_time_period_representation_mock(mock_csv):
    mock_data = DataFrame(columns=["Id_1", "Id_2"])
    dataset = Dataset(
        name="test_dataset",
        components={
            "Id_1": Component(
                name="Id_1",
                data_type=DataTypes.Integer,
                role=Role.IDENTIFIER,
                nullable=False,
            ),
            "Id_2": Component(
                name="Id_2",
                data_type=DataTypes.TimePeriod,
                role=Role.IDENTIFIER,
                nullable=False,
            ),
        },
        data=mock_data,
    )
    output_path = "s3://path/to/output/"

    save_datapoints(TimePeriodRepresentation.VTL, dataset, output_path)

    expected_path = "s3://path/to/output/test_dataset.csv"
    mock_csv.assert_called_once_with(expected_path, index=False)


@pytest.mark.parametrize("dataset, reference", params)
def test_save_datapoints(dataset, reference, tmp_path_factory):
    output_path = tmp_path_factory.mktemp("test")
    save_datapoints(None, dataset, output_path=output_path)
    result = pd.read_csv(output_path / f"{dataset.name}.csv")
    pd.testing.assert_frame_equal(result, dataset.data)


@patch("pandas.read_csv")
def test_load_datapoints_s3(mock_read_csv):
    input_path = "s3://path/to/input/dataset.csv"
    load_datapoints(components={}, dataset_name="dataset", csv_path=input_path)
    mock_read_csv.assert_called_once_with(
        input_path,
        dtype={},
        engine="c",
        keep_default_na=False,
        na_values=[""],
        encoding_errors="replace",
    )


@patch("pandas.read_csv")
def test_run_s3(mock_read_csv):
    with open(filepath_datastructure / "DS_1.json") as f:
        data_structures = json.load(f)

    input_path = "s3://path/to/input/DS_1.csv"
    with pytest.raises(InputValidationException):
        run(script="DS_r := DS_1;", data_structures=data_structures, datapoints=input_path)

    dtypes = {comp["name"]: np.object_ for comp in data_structures["datasets"][0]["DataStructure"]}
    mock_read_csv.assert_called_once_with(
        input_path,
        dtype=dtypes,
        engine="c",
        keep_default_na=False,
        na_values=[""],
        encoding_errors="replace",
    )
