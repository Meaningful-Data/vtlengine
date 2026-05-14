from typing import Union

from vtlengine.DataTypes import Date
from vtlengine.Model import Dataset, Scalar


def _space_to_t(value: str) -> str:
    if len(value) > 10 and value[10] == " ":
        return value[:10] + "T" + value[11:]
    return value


def format_date_iso8601(operand: Union[Dataset, Scalar]) -> None:
    """Convert internal Date representation (space separator) to ISO 8601 (T separator)."""
    if isinstance(operand, Scalar):
        if operand.data_type == Date and isinstance(operand.value, str) and len(operand.value) > 10:
            operand.value = _space_to_t(operand.value)
        return
    if operand.data is None or len(operand.data) == 0:
        return
    for comp in operand.components.values():
        if comp.data_type == Date:
            operand.data[comp.name] = (
                operand.data[comp.name]
                .map(_space_to_t, na_action="ignore")
                .astype("string[pyarrow]")
            )
