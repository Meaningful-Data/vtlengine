from pathlib import Path
from typing import Optional, Union

from _duckdb import Error
from duckdb import DuckDBPyRelation  # type: ignore[import-untyped]

from vtlengine.duckdb.duckdb_utils import empty_relation
from vtlengine.Exceptions import RunTimeError
from vtlengine.files.output._time_period_representation import (
    TimePeriodRepresentation,
    format_time_period_external_representation,
)
from vtlengine.Model import Dataset


def save_datapoints(
    time_period_representation: Optional[TimePeriodRepresentation],
    dataset: Dataset,
    output_path: Union[str, Path],
) -> None:
    if dataset.data is None:
        dataset.data = empty_relation(dataset.get_components_names())
    if time_period_representation is not None:
        format_time_period_external_representation(dataset, time_period_representation)
    if isinstance(dataset.data, DuckDBPyRelation):
        try:
            dataset.data = dataset.data.df()
        except Error as e:
            raise RunTimeError.map_duckdb_error(e)
    if isinstance(output_path, str):
        # __check_s3_extra()
        if "__index__" in dataset.data.columns:
            dataset.data = dataset.data.drop(columns="__index__")

        if output_path.endswith("/"):
            s3_file_output = output_path + f"{dataset.name}.csv"
        else:
            s3_file_output = output_path + f"/{dataset.name}.csv"
        # start = time()
        dataset.data.df().to_csv(s3_file_output, index=False)
        # end = time()
        # print(f"Dataset {dataset.name} saved to {s3_file_output}")
        # print(f"Time to save data on s3 URI: {end - start}")
    else:
        dataset.data.df().to_csv(output_path / f"{dataset.name}.csv", index=False)
