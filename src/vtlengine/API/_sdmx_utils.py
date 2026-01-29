"""
SDMX utility functions for the VTL Engine API.

This module contains helper functions for handling SDMX mappings and conversions
between SDMX URNs and VTL dataset names.
"""

import warnings
from typing import Dict, List, Optional, Sequence, Union

from pysdmx.io.pd import PandasDataset
from pysdmx.model import DataflowRef, Reference
from pysdmx.model.dataflow import Dataflow, Schema
from pysdmx.model.vtl import VtlDataflowMapping
from pysdmx.util import parse_urn

from vtlengine.Exceptions import InputValidationException


def _convert_vtl_dataflow_mapping(mapping: VtlDataflowMapping) -> Dict[str, str]:
    """
    Convert a VtlDataflowMapping object to a dict mapping SDMX URN to VTL dataset name.

    Args:
        mapping: VtlDataflowMapping object to convert.

    Returns:
        Dict with single entry mapping short_urn -> dataflow_alias.

    Raises:
        InputValidationException: If dataflow type is invalid.
    """
    if mapping.to_vtl_mapping_method is not None:
        warnings.warn(
            "To_vtl_mapping_method is not implemented yet, we will use the Basic "
            "method with old data."
        )
    if mapping.from_vtl_mapping_method is not None:
        warnings.warn(
            "From_vtl_mapping_method is not implemented yet, we will use the Basic "
            "method with old data."
        )

    if isinstance(mapping.dataflow, str):
        short_urn = str(parse_urn(mapping.dataflow))
    elif isinstance(mapping.dataflow, (Reference, DataflowRef)):
        short_urn = str(mapping.dataflow)
    elif isinstance(mapping.dataflow, Dataflow):
        short_urn = mapping.dataflow.short_urn
    else:
        raise InputValidationException(
            "Expected str, Reference, DataflowRef or Dataflow type for dataflow in "
            "VtlDataflowMapping."
        )
    return {short_urn: mapping.dataflow_alias}


def _convert_sdmx_mappings(
    mappings: Optional[Union[VtlDataflowMapping, Dict[str, str]]],
) -> Optional[Dict[str, str]]:
    """
    Convert sdmx_mappings parameter to dict format for internal use.

    Args:
        mappings: None, dict, or VtlDataflowMapping object.

    Returns:
        None if mappings is None, otherwise dict mapping SDMX URNs to VTL dataset names.

    Raises:
        InputValidationException: If mappings type is invalid.
    """
    if mappings is None:
        return None
    if isinstance(mappings, dict):
        return mappings
    if isinstance(mappings, VtlDataflowMapping):
        return _convert_vtl_dataflow_mapping(mappings)
    raise InputValidationException("Expected dict or VtlDataflowMapping type for mappings.")


def _build_mapping_dict(
    datasets: Sequence[PandasDataset],
    mappings: Optional[Union[VtlDataflowMapping, Dict[str, str]]],
    input_names: List[str],
) -> Dict[str, str]:
    """
    Build mapping dict from SDMX URNs to VTL dataset names.

    Args:
        datasets: Sequence of PandasDataset objects.
        mappings: Optional mapping configuration (None, dict, or VtlDataflowMapping).
        input_names: List of input dataset names extracted from the VTL script.

    Returns:
        Dict mapping short_urn -> vtl_dataset_name.

    Raises:
        InputValidationException: If mapping configuration is invalid.
    """
    if mappings is None:
        if len(datasets) != 1:
            raise InputValidationException(code="0-1-3-3")
        if len(input_names) != 1:
            raise InputValidationException(code="0-1-3-1", number_datasets=len(input_names))
        schema = datasets[0].structure
        if not isinstance(schema, Schema):
            raise InputValidationException(code="0-1-3-2", schema=schema)
        return {schema.short_urn: input_names[0]}

    if isinstance(mappings, dict):
        return mappings

    if isinstance(mappings, VtlDataflowMapping):
        return _convert_vtl_dataflow_mapping(mappings)

    raise InputValidationException("Expected dict or VtlDataflowMapping type for mappings.")
