from pathlib import Path
from typing import Optional, Union

import pandas as pd

from vtlengine.__extras_check import __check_s3_extra
from vtlengine.files.output._time_period_representation import (
    TimePeriodRepresentation,
    format_time_period_external_representation,
)
from vtlengine.Model import Dataset
from vtlengine.Utils._number_config import get_float_format


def save_datapoints(
    time_period_representation: Optional[TimePeriodRepresentation],
    dataset: Dataset,
    output_path: Union[str, Path],
) -> None:
    if dataset.data is None:
        dataset.data = pd.DataFrame()
    if time_period_representation is not None:
        format_time_period_external_representation(dataset, time_period_representation)

    # Get float format based on environment configuration
    float_format = get_float_format()

    if isinstance(output_path, str):
        __check_s3_extra()
        if output_path.endswith("/"):
            s3_file_output = output_path + f"{dataset.name}.csv"
        else:
            s3_file_output = output_path + f"/{dataset.name}.csv"
        # start = time()
        dataset.data.to_csv(s3_file_output, index=False, float_format=float_format)
        # end = time()
        # print(f"Dataset {dataset.name} saved to {s3_file_output}")
        # print(f"Time to save data on s3 URI: {end - start}")
    else:
        output_file = output_path / f"{dataset.name}.csv"
        dataset.data.to_csv(output_file, index=False, float_format=float_format)
