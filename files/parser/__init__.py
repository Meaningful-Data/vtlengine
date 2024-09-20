from csv import DictReader
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np

from DataTypes import Date, TimePeriod, TimeInterval, Integer, Number, Boolean, Duration, \
    SCALAR_TYPES_CLASS_REVERSE
from DataTypes.TimeHandling import DURATION_MAPPING
from Exceptions import InputValidationException, SemanticError
from Model import Component, Role
from files.parser._rfc_dialect import register_rfc
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
    register_rfc()
    try:
        with open(csv_path, 'r') as f:
            reader = DictReader(f, dialect='rfc')
            csv_columns = reader.fieldnames

    except UnicodeDecodeError as error:
        # https://coderwall.com/p/stzy9w/raising-unicodeencodeerror-and-unicodedecodeerror-
        # manually-for-testing-purposes
        raise InputValidationException("0-1-2-5", file=csv_path.name) from error
    except InputValidationException as ie:
        raise InputValidationException("{}".format(str(ie))) from None
    except Exception as e:
        raise InputValidationException(
            f"ERROR: {str(e)}, review file {str(csv_path.as_posix())}"
        ) from None

    if not csv_columns:
        raise InputValidationException(code='0-1-1-6', file=csv_path)

    if len(list(set(csv_columns))) != len(csv_columns):
        duplicates = list(set([item for item in csv_columns if csv_columns.count(item) > 1]))
        raise Exception(f"Duplicated columns {', '.join(duplicates)} found in file.")

    comp_names = set([c.name for c in components.values() if c.role == Role.IDENTIFIER])
    comps_missing = [id_m for id_m in comp_names if id_m not in reader.fieldnames]
    if comps_missing:
        comps_missing = ", ".join(comps_missing)
        raise InputValidationException(code='0-1-1-8', ids=comps_missing, file=str(csv_path.name))


def _pandas_load_csv(components: Dict[str, Component], csv_path: Path) -> pd.DataFrame:
    obj_dtypes = {comp_name: np.object_ for comp_name, comp in components.items()}

    register_rfc()

    try:
        data = pd.read_csv(csv_path, dialect='rfc', dtype=obj_dtypes, engine='c',
                           keep_default_na=False,
                           na_values=[''])
    except UnicodeDecodeError as error:
        raise InputValidationException(code="0-1-2-5", file=csv_path.name)
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
            data = data[data["ACTION"] != "D"]
            data.drop(columns=["ACTION"], inplace=True)

    # Validate identifiers
    comp_names = set([c.name for c in components.values() if c.role == Role.IDENTIFIER])
    comps_missing = [id_m for id_m in comp_names if id_m not in data.columns]
    if comps_missing:
        comps_missing = ", ".join(comps_missing)
        raise InputValidationException(code='0-1-1-7', ids=comps_missing, file=str(csv_path.name))

    # Fill rest of components with null values
    for comp_name, comp in components.items():
        if comp_name not in data:
            if not comp.nullable:
                raise InputValidationException(f"Component {comp_name} is missing in the file.")
            data[comp_name] = None
    return data

def _parse_boolean(value: str):
    if value.lower() == "true" or value.lower() == "1":
        return True
    return False

def _validate_pandas(components: Dict[str, Component], data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    # Identifier checking
    id_names = [comp_name for comp_name, comp in components.items() if comp.role == Role.IDENTIFIER]

    for id_name in id_names:
        if data[id_name].isnull().any():
            raise SemanticError("0-1-1-4", null_identifier=id_name, name=dataset_name)

    if len(id_names) == 0 and len(data) > 1:
        raise SemanticError("0-1-1-5", name=dataset_name)

    data = data.fillna(np.nan).replace([np.nan], [None])
    # Checking data types on all data types
    comp_name = ""
    comp = None
    try:

        for comp_name, comp in components.items():
            if comp.data_type in (Date, TimePeriod, TimeInterval):
                data[comp_name] = data[comp_name].map(TIME_CHECKS_MAPPING[comp.data_type],
                                                      na_action='ignore')
            elif comp.data_type == Integer:
                data[comp_name] = data[comp_name].map(lambda x: Integer.cast(float(x)), na_action='ignore')
            elif comp.data_type == Number:
                data[comp_name] = data[comp_name].map(lambda x: float(x), na_action='ignore')
            elif comp.data_type == Boolean:
                data[comp_name] = data[comp_name].map(lambda x: _parse_boolean(x), na_action='ignore')
            elif comp.data_type == Duration:
                values_correct = data[comp_name].map(lambda x: x.replace(" ", "") in DURATION_MAPPING, na_action='ignore').all()
                if not values_correct:
                    raise ValueError(f"Duration values are not correct in column {comp_name}")
            else:
                data[comp_name] = data[comp_name].map(lambda x: str(x).replace('"', ''),
                                                      na_action='ignore')
            data[comp_name] = data[comp_name].astype(np.object_, errors='raise')
    except ValueError as e:
        str_comp = SCALAR_TYPES_CLASS_REVERSE[comp.data_type]
        raise SemanticError("0-1-1-12", name=dataset_name, column=comp_name, type=str_comp)

    return data


def load_datapoints(components: Dict[str, Component],
                    dataset_name: str,
                    csv_path: Optional[Path] = None):
    if csv_path is None or not csv_path.exists():
        return pd.DataFrame(columns=list(components.keys()))

    _validate_csv_path(components, csv_path)
    data = _pandas_load_csv(components, csv_path)
    data = _validate_pandas(components, data, dataset_name)

    return data
