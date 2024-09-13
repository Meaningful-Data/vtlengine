from enum import Enum
from typing import Dict

from DataTypes import TimePeriod
from DataTypes.TimeHandling import TimePeriodHandler
from Model import Dataset, Scalar


class TimePeriodRepresentation(Enum):
    # Time Period output format
    SDMX_GREGORIAN = 'sdmx_gregorian'
    SDMX_REPORTING = 'sdmx_reporting'
    VTL = 'vtl'

    @classmethod
    def check_value(cls, value: str):
        if value not in cls._value2member_map_:
            raise Exception("Invalid Time Period Representation")
        return cls(value)


def _format_vtl_representation(value: str):
    return TimePeriodHandler(value).vtl_representation()


def format_time_period_external_representation(datasets: Dict[str, Dataset | Scalar],
                                               mode: TimePeriodRepresentation):
    """
    From SDMX time period representation to standard VTL representation (no hyphen).
    'A': 'nothing to do',
    'S': 'YYYY-Sx',
    'Q': 'YYYY-Qx',
    'M': 'YYYY-MM',
    'W': 'YYYY-Wxx',
    'D': 'YYYY-MM-DD'
    """
    if mode == TimePeriodRepresentation.SDMX_REPORTING:
        return datasets
    elif mode == TimePeriodRepresentation.SDMX_GREGORIAN:
        raise NotImplementedError

    # VTL Representation
    for dataset in datasets.values():
        if isinstance(dataset, Dataset):
            if dataset.data is None or len(dataset.data) == 0:
                continue
            for comp in dataset.components.values():
                if comp.data_type == TimePeriod:
                    dataset.data[comp.name] = dataset.data[comp.name].map(
                        _format_vtl_representation,
                        na_action='ignore')

    return datasets
