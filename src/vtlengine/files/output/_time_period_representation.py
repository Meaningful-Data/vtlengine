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


def _format_sdmx_gregorian_representation(value: str) -> str:
    return TimePeriodHandler(value).sdmx_gregorian_representation()


def _format_sdmx_reporting_representation(value: str) -> str:
    return TimePeriodHandler(value).sdmx_reporting_representation()


def format_time_period_external_representation(
    dataset: Union[Dataset, Scalar], mode: TimePeriodRepresentation
) -> None:
    """
    Converts internal time period representation to the requested external format.

    SDMX Reporting: YYYY-A1, YYYY-Ss, YYYY-Qq, YYYY-Mmm, YYYY-Www, YYYY-Dddd
    SDMX Gregorian: YYYY, YYYY-MM, YYYY-MM-DD (only A, M, D supported)
    VTL: YYYY, YYYYSn, YYYYQn, YYYYMm, YYYYWw, YYYYDd (no hyphens)
    """
    if isinstance(dataset, Scalar):
        if dataset.data_type != TimePeriod or dataset.value is None:
            return
        
        value = dataset.value
        if mode == TimePeriodRepresentation.VTL:
            dataset.value = _format_vtl_representation(value)
        elif mode == TimePeriodRepresentation.SDMX_GREGORIAN:
            dataset.value = _format_sdmx_gregorian_representation(value)
        elif mode == TimePeriodRepresentation.SDMX_REPORTING:
            dataset.value = _format_sdmx_reporting_representation(value)
        return

    if dataset.data is None or len(dataset.data) == 0:
        return
    if mode == TimePeriodRepresentation.VTL:
        formatter = _format_vtl_representation
    elif mode == TimePeriodRepresentation.SDMX_GREGORIAN:
        formatter = _format_sdmx_gregorian_representation
    elif mode == TimePeriodRepresentation.SDMX_REPORTING:
        formatter = _format_sdmx_reporting_representation

    for comp in dataset.components.values():
        if comp.data_type == TimePeriod:
            dataset.data[comp.name] = dataset.data[comp.name].map(formatter, na_action="ignore")
