from pathlib import Path
from typing import Optional, Union

import duckdb
import pandas as pd
from duckdb.duckdb import DuckDBPyRelation  # type: ignore[import-untyped]

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
        dataset.data = pd.DataFrame()
    if time_period_representation is not None:
        format_time_period_external_representation(dataset, time_period_representation)
    if isinstance(dataset.data, DuckDBPyRelation):
        try:
            dataset.data = dataset.data.df()
        except duckdb.Error as e:
            raise RunTimeError.map_duckdb_error(e)
    if isinstance(output_path, str):
        # __check_s3_extra()
        base_output = output_path if output_path.endswith("/") else output_path + "/"
    else:
        base_output = Path(output_path)  # type: ignore[assignment]
        # start = time()
        # end = time()
        # print(f"Dataset {dataset.name} saved to {s3_file_output}")
        # print(f"Time to save data on s3 URI: {end - start}")
    if str(output_path).lower().endswith(".parquet"):
        file_output = (
            base_output if isinstance(base_output, str) else base_output / f"{dataset.name}.parquet"  # type: ignore[redundant-expr]
        )
        dataset.data.to_parquet(file_output, index=False)
    else:
        file_output = (
            base_output if isinstance(base_output, str) else base_output / f"{dataset.name}.csv"  # type: ignore[redundant-expr]
        )
        dataset.data.to_csv(file_output, index=False)
