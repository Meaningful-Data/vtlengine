from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from vtlengine import DataTypes
from vtlengine.files.output import TimePeriodRepresentation, save_datapoints
from vtlengine.Model import Component, Dataset, Role

base_path = Path(__file__).parent
filepath_output = base_path / "data" / "DataSet" / "output"

params = [
    (
        Dataset(
            name="test_dataset",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=DataTypes.String, role=Role.IDENTIFIER, nullable=False
                ),
            },
            data=pd.DataFrame(columns=["Id_1", "Id_2"]),
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
                name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
            ),
            "Id_2": Component(
                name="Id_2", data_type=DataTypes.String, role=Role.IDENTIFIER, nullable=False
            ),
        },
        data=None,
    )
    output_path = "path/to/output"

    save_datapoints(None, dataset, output_path)

    expected_path = "path/to/output/test_dataset.csv"
    mock_csv.assert_called_once_with(expected_path, index=False)


@patch("pandas.DataFrame.to_csv")
def test_save_datapoints_with_data_mock(mock_csv):
    mock_data = pd.DataFrame(columns=["Id_1", "Id_2"])
    dataset = Dataset(
        name="test_dataset",
        components={
            "Id_1": Component(
                name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
            ),
            "Id_2": Component(
                name="Id_2", data_type=DataTypes.String, role=Role.IDENTIFIER, nullable=False
            ),
        },
        data=mock_data,
    )
    output_path = "path/to/output/"

    save_datapoints(None, dataset, output_path)

    expected_path = "path/to/output/test_dataset.csv"
    mock_csv.assert_called_once_with(expected_path, index=False)


@patch("pandas.DataFrame.to_csv")
def test_save_datapoints_with_data_and_time_period_representation_mock(mock_csv):
    mock_data = pd.DataFrame(columns=["Id_1", "Id_2"])
    dataset = Dataset(
        name="test_dataset",
        components={
            "Id_1": Component(
                name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
            ),
            "Id_2": Component(
                name="Id_2", data_type=DataTypes.TimePeriod, role=Role.IDENTIFIER, nullable=False
            ),
        },
        data=mock_data,
    )
    output_path = "path/to/output/"

    save_datapoints(TimePeriodRepresentation.VTL, dataset, output_path)

    expected_path = "path/to/output/test_dataset.csv"
    mock_csv.assert_called_once_with(expected_path, index=False)


@pytest.mark.parametrize("dataset, reference", params)
def test_save_datapoints(dataset, reference, tmp_path_factory):
    output_path = tmp_path_factory.mktemp("test")
    save_datapoints(None, dataset, output_path=output_path)
    result = pd.read_csv(output_path / f"{dataset.name}.csv")
    pd.testing.assert_frame_equal(result, dataset.data)

