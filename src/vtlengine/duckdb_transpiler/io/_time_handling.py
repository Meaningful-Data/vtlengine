"""
Time period representation handling for DuckDB results.

Applies output format conversion (VTL, SDMX Reporting, SDMX Gregorian, Natural)
to TimePeriod columns using DuckDB SQL macros on the existing connection.
"""

from typing import Dict, Optional

import duckdb

from vtlengine.DataTypes import TimePeriod
from vtlengine.files.output._time_period_representation import (
    TimePeriodRepresentation,
    format_time_period_external_representation,
)
from vtlengine.Model import Dataset, Scalar

_REPR_MACRO: Dict[TimePeriodRepresentation, str] = {
    TimePeriodRepresentation.VTL: "vtl_period_to_vtl",
    TimePeriodRepresentation.SDMX_REPORTING: "vtl_period_to_sdmx_reporting",
    TimePeriodRepresentation.SDMX_GREGORIAN: "vtl_period_to_sdmx_gregorian",
    TimePeriodRepresentation.NATURAL: "vtl_period_to_natural",
}


def apply_time_period_representation(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    output_datasets: Dict[str, Dataset],
    output_scalars: Dict[str, Scalar],
    representation: Optional[TimePeriodRepresentation],
) -> None:
    """Apply time period output representation to a DuckDB table in-place.

    Uses UPDATE to convert internal canonical format to the requested format
    directly on the existing connection. Called before saving to CSV or
    fetching as DataFrame.

    Scalars are skipped here — they are formatted after fetching via
    ``format_time_period_scalar``.
    """
    if representation is None:
        return

    # Skip scalars — handled after fetch via format_time_period_scalar
    if table_name in output_scalars:
        return

    # Dataset: find TimePeriod columns and apply macro via UPDATE
    ds = output_datasets.get(table_name)
    if ds is None or not ds.components:
        return

    tp_cols = [c.name for c in ds.components.values() if c.data_type == TimePeriod]
    if not tp_cols:
        return

    macro = _REPR_MACRO[representation]
    set_clauses = ", ".join(f'"{col}" = {macro}("{col}")' for col in tp_cols)
    where_clauses = " OR ".join(f'"{col}" IS NOT NULL' for col in tp_cols)
    conn.execute(f'UPDATE "{table_name}" SET {set_clauses} WHERE {where_clauses}')


def format_time_period_scalar(
    scalar: Scalar,
    representation: Optional[TimePeriodRepresentation],
) -> None:
    """Apply time period output representation to a Scalar value."""
    if representation is None:
        return
    format_time_period_external_representation(scalar, representation)
