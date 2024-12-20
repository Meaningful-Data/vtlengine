import boto3
import pandas as pd
from moto import mock_aws

from vtlengine import DataTypes
from vtlengine.Model import Dataset, Role, Component
from vtlengine.files.output import save_datapoints
from vtlengine.files.output._time_period_representation import TimePeriodRepresentation


@mock_aws
def test_save_datapoints_with_data():
    s3 = boto3.resource("s3", region_name="us-east-1")
    bucket_name = 'my_bucket'
    s3.create_bucket(Bucket=bucket_name)
    data = pd.DataFrame(columns=["Id_1", "Id_2"])
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
        data=data)

    output_path = f"s3://{bucket_name}/test-folder/"

    save_datapoints(None, dataset, output_path)

    s3_response = s3.Object(bucket_name, "test-folder/test_dataset.csv").get()['Body'].read().decode("utf-8")
    assert s3_response == data.to_csv()


# @patch('pandas.DataFrame.to_csv')
# def test_save_datapoints_without_data(mock_csv):
#     dataset = Dataset(name='test_dataset', components={
#         "Id_1": Component(
#             name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
#         ),
#         "Id_2": Component(
#             name="Id_2", data_type=DataTypes.String, role=Role.IDENTIFIER, nullable=False
#         ),
#     },
#                       data=None)
#     output_path = 'path/to/output'
#
#     save_datapoints(None, dataset, output_path)
#
#     expected_path = 'path/to/output/test_dataset.csv'
#     mock_csv.assert_called_once_with(expected_path, index=False)
