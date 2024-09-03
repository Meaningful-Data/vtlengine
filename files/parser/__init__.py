from csv import DictReader
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np

from DataTypes import Date, TimePeriod, TimeInterval, Integer, Number, Boolean
from Exceptions import InputValidationException
from Model import Component, Role
from files.parser._time_checking import check_date, check_time_period, check_time

TIME_CHECKS_MAPPING = {
    Date: check_date,
    TimePeriod: check_time_period,
    TimeInterval: check_time
}


def _validate_csv_path(components: Dict[str, Component], csv_path: Path):
    # GE1 check if the file is empty
    if not csv_path.exists():
        raise Exception(f"Path {csv_path} does not exist.")
    if not csv_path.is_file():
        raise Exception(f"Path {csv_path} is not a file.")
    try:
        with open(csv_path, 'r') as f:
            reader = DictReader(f)
            csv_columns = reader.fieldnames

    except UnicodeDecodeError as error:
        # https://coderwall.com/p/stzy9w/raising-unicodeencodeerror-and-unicodedecodeerror-
        # manually-for-testing-purposes
        error_message = f"The file {csv_path.name} is not utf-8 encoded."
        raise UnicodeDecodeError("utf-8", b"", error.start, error.end, error_message) from None
    except InputValidationException as ie:
        raise InputValidationException("{}".format(str(ie))) from None
    except Exception as e:
        raise InputValidationException(
            f"ERROR: {str(e)}, review file {str(csv_path.as_posix())}"
        ) from None

    if not csv_columns:
        raise InputValidationException(code='0-1-1-6', file=csv_path)

    if len(list(set(csv_columns))) != len(csv_columns):
        raise Exception("Duplicated columns found in the file.")

    comp_names = set([c.name for c in components.values() if c.role == Role.IDENTIFIER])
    comps_missing = [id_m for id_m in comp_names if id_m not in reader.fieldnames]
    if comps_missing:
        comps_missing = ", ".join(comps_missing)
        raise InputValidationException(code='0-1-1-7', ids=comps_missing, file=str(csv_path.name))


def _pandas_load_csv(components: Dict[str, Component], csv_path: Path) -> pd.DataFrame:
    obj_dtypes = {comp_name: np.object_ for comp_name, comp in components.items()}

    data = pd.read_csv(csv_path, dtype=obj_dtypes, engine='python',
                       keep_default_na=False,
                       na_values=[''], encoding='utf-8')
    # Fast loading from SDMX-CSV
    if "DATAFLOW" in data.columns and data.columns[0] == "DATAFLOW":
        if "DATAFLOW" not in components:
            data.drop(columns=["DATAFLOW"], inplace=True)
    if "STRUCTURE" in data.columns and data.columns[0] == "STRUCTURE":
        if "STRUCTURE" not in components:
            data.drop(columns=["STRUCTURE"], inplace=True)
        if "STRUCTURE_ID" in data.columns:
            data.drop(columns=["STRUCTURE_ID"], inplace=True)
        if "ACTION" in data.columns:
            dfDataPoints = data[data["ACTION"] != "D"]
            dfDataPoints.drop(columns=["ACTION"], inplace=True)

    # Validate identifiers
    comp_names = set([c.name for c in components.values() if c.role == Role.IDENTIFIER])
    comps_missing = [id_m for id_m in comp_names if id_m not in data.columns]
    if comps_missing:
        comps_missing = ", ".join(comps_missing)
        raise InputValidationException(code='0-1-1-7', ids=comps_missing, file=str(csv_path.name))

    # Fill rest of components with null values
    for comp_name in components:
        if comp_name not in data:
            data[comp_name] = None
    return data

def _parse_boolean(value: str):
    if value.lower() == "true" or value.lower() == "1":
        return True
    return False

def _validate_pandas(components: Dict[str, Component], data: pd.DataFrame):
    # Identifier checking
    id_names = [comp_name for comp_name, comp in components.items() if comp.role == Role.IDENTIFIER]

    for id_name in id_names:
        if data[id_name].isnull().any():
            raise Exception(f"Identifiers cannot have null values, check column {id_name}")

    data = data.fillna(np.nan).replace([np.nan], [None])
    # Checking time data types on all components
    for comp_name, comp in components.items():
        if comp.data_type in (Date, TimePeriod, TimeInterval):
            data[comp_name] = data[comp_name].map(TIME_CHECKS_MAPPING[comp.data_type],
                                                  na_action='ignore')

    # We only modify the measure and attribute values, rest we keep as object
    non_identifiers = {c.name: c for c in components.values()
                      if c.role in (Role.MEASURE, Role.ATTRIBUTE)}
    for comp_name, comp in non_identifiers.items():
        if comp.data_type == Integer:
            data[comp_name] = data[comp_name].map(lambda x: int(float(x)), na_action='ignore')
        elif comp.data_type == Number:
            data[comp_name] = data[comp_name].map(lambda x: float(x), na_action='ignore')
        elif comp.data_type == Boolean:
            data[comp_name] = data[comp_name].map(lambda x: _parse_boolean(x), na_action='ignore')
        data[comp_name] = data[comp_name].astype(np.object_, errors='raise')

    return data


def load_datapoints(components: Dict[str, Component], csv_path: Optional[Path] = None):
    if csv_path is None or not csv_path.exists():
        return pd.DataFrame(columns=list(components.keys()))

    _validate_csv_path(components, csv_path)
    data = _pandas_load_csv(components, csv_path)
    data = _validate_pandas(components, data)

    return data
