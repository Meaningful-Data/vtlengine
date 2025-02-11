from enum import Enum
from typing import Union

from vtlengine.DataTypes import TimePeriod
from vtlengine.DataTypes.TimeHandling import TimePeriodHandler
from vtlengine.Model import Dataset, Scalar


class TimePeriodRepresentation(Enum):
    # Time Period output format
    SDMX_GREGORIAN = "sdmx_gregorian"
    SDMX_REPORTING = "sdmx_reporting"
    VTL = "vtl"

    @classmethod
    def check_value(cls, value: str) -> "TimePeriodRepresentation":
        if value not in cls._value2member_map_:
            raise Exception("Invalid Time Period Representation")
        return cls(value)


def _format_vtl_representation(value: str) -> str:
    return TimePeriodHandler(value).vtl_representation()


def format_time_period_external_representation(
    dataset: Union[Dataset, Scalar], mode: TimePeriodRepresentation
) -> None:
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
        return
    elif mode == TimePeriodRepresentation.SDMX_GREGORIAN:
        raise NotImplementedError

    if isinstance(dataset, Scalar):
        return

    # VTL Representation
    if dataset.data is None or len(dataset.data) == 0:
        return
    for comp in dataset.components.values():
        if comp.data_type == TimePeriod:
            dataset.data[comp.name] = dataset.data[comp.name].map(
                _format_vtl_representation, na_action="ignore"
            )

    return
