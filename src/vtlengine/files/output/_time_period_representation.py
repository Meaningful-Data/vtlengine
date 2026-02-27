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
    LEGACY = "legacy"

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


def _format_legacy_representation(value: str) -> str:
    return TimePeriodHandler(value).legacy_representation()


def format_time_period_external_representation(
    operand: Union[Dataset, Scalar], mode: TimePeriodRepresentation
) -> None:
    """
    Converts internal time period representation to the requested external format.

    SDMX Reporting: YYYY-A1, YYYY-Ss, YYYY-Qq, YYYY-Mmm, YYYY-Www, YYYY-Dddd
    SDMX Gregorian: YYYY, YYYY-MM, YYYY-MM-DD (only A, M, D supported)
    VTL: YYYY, YYYYSn, YYYYQn, YYYYMm, YYYYWw, YYYYDd (no hyphens)
    Legacy: YYYY, YYYY-Sx, YYYY-Qx, YYYY-Mxx, YYYY-Wxx, YYYY-MM-DD
    """
    if isinstance(operand, Scalar):
        if operand.data_type != TimePeriod or operand.value is None:
            return

        value = operand.value
        if mode == TimePeriodRepresentation.VTL:
            operand.value = _format_vtl_representation(value)
        elif mode == TimePeriodRepresentation.SDMX_GREGORIAN:
            operand.value = _format_sdmx_gregorian_representation(value)
        elif mode == TimePeriodRepresentation.SDMX_REPORTING:
            operand.value = _format_sdmx_reporting_representation(value)
        elif mode == TimePeriodRepresentation.LEGACY:
            operand.value = _format_legacy_representation(value)
        return

    if operand.data is None or len(operand.data) == 0:
        return
    if mode == TimePeriodRepresentation.VTL:
        formatter = _format_vtl_representation
    elif mode == TimePeriodRepresentation.SDMX_GREGORIAN:
        formatter = _format_sdmx_gregorian_representation
    elif mode == TimePeriodRepresentation.SDMX_REPORTING:
        formatter = _format_sdmx_reporting_representation
    elif mode == TimePeriodRepresentation.LEGACY:
        formatter = _format_legacy_representation

    for comp in operand.components.values():
        if comp.data_type == TimePeriod:
            operand.data[comp.name] = (
                operand.data[comp.name].map(formatter, na_action="ignore").astype("string[pyarrow]")
            )
