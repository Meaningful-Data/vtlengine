"""
SDMX utility functions for the VTL Engine API.

This module contains helper functions for handling SDMX mappings and conversions
between SDMX URNs and VTL dataset names.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from pysdmx.io.pd import PandasDataset
from pysdmx.model import DataflowRef, Reference
from pysdmx.model.dataflow import Dataflow, DataStructureDefinition, Schema
from pysdmx.model.vtl import VtlDataflowMapping, VtlMappingScheme
from pysdmx.util import parse_urn

from vtlengine.Exceptions import InputValidationException

# Any supported form of the ``mappings`` / ``sdmx_mappings`` argument. A single SDMX
# dataflow may feed several VTL datasets, so dict values may be a single name or a list.
MappingsInput = Union[
    VtlDataflowMapping,
    Sequence[VtlDataflowMapping],
    VtlMappingScheme,
    Dict[str, Union[str, List[str]]],
]

# Extra ``data_structures`` / ``datapoints`` accepted by ``run_sdmx`` (mirrors ``run``).
DataStructureItem = Union[Dict[str, Any], Path, str, Schema, DataStructureDefinition, Dataflow]
DataStructuresInput = Union[DataStructureItem, List[DataStructureItem]]
DatapointsInput = Union[
    Dict[str, Union[pd.DataFrame, str, Path]], List[Union[str, Path]], str, Path
]


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


def _normalize_mappings(
    mappings: Optional[MappingsInput],
) -> Optional[Dict[str, List[str]]]:
    """
    Normalize any supported mappings input into ``{short_urn: [vtl_name, ...]}``.

    A single SDMX dataflow may be mapped to several VTL dataset names, which is why the
    values are always lists. Accepts a plain dict (values may be a single name or a list
    of names), a single ``VtlDataflowMapping``, a sequence of ``VtlDataflowMapping``
    objects, or a ``VtlMappingScheme``.

    Args:
        mappings: None, dict, VtlDataflowMapping, sequence of VtlDataflowMapping, or
            VtlMappingScheme.

    Returns:
        None if mappings is None, otherwise a dict mapping each SDMX short-URN to the
        list of VTL dataset names it feeds.

    Raises:
        InputValidationException: If mappings (or one of its items) has an invalid type.
    """
    if mappings is None:
        return None
    if isinstance(mappings, dict):
        return {
            urn: [names] if isinstance(names, str) else list(names)
            for urn, names in mappings.items()
        }
    if isinstance(mappings, VtlDataflowMapping):
        items: Sequence[VtlDataflowMapping] = [mappings]
    elif isinstance(mappings, VtlMappingScheme):
        items = [m for m in mappings.items if isinstance(m, VtlDataflowMapping)]
    elif isinstance(mappings, (list, tuple)):
        items = mappings
    else:
        raise InputValidationException("Expected dict or VtlDataflowMapping type for mappings.")
    result: Dict[str, List[str]] = {}
    for mapping in items:
        if not isinstance(mapping, VtlDataflowMapping):
            raise InputValidationException("Expected dict or VtlDataflowMapping type for mappings.")
        for urn, alias in _convert_vtl_dataflow_mapping(mapping).items():
            result.setdefault(urn, []).append(alias)
    return result


def _convert_sdmx_mappings(
    mappings: Optional[MappingsInput],
) -> Optional[Dict[str, str]]:
    """
    Convert sdmx_mappings parameter to the single-name dict used by the shared
    structure-loading path (``run``/``semantic_analysis``).

    That path resolves one SDMX structure to exactly one dataset name. Mapping several
    VTL datasets to a single structure is only possible through :func:`_build_mapping_dict`
    inside ``run_sdmx``; here it is rejected so the caller gets a clear error instead of a
    silently dropped alias.

    Args:
        mappings: None, dict, VtlDataflowMapping, sequence of VtlDataflowMapping, or
            VtlMappingScheme.

    Returns:
        None if mappings is None, otherwise dict mapping SDMX URNs to VTL dataset names.

    Raises:
        InputValidationException: If mappings type is invalid (or a single structure is
            mapped to more than one VTL dataset name, ``0-1-3-13``).
    """
    normalized = _normalize_mappings(mappings)
    if normalized is None:
        return None
    for urn, names in normalized.items():
        if len(names) > 1:
            raise InputValidationException(code="0-1-3-13", short_urn=urn, names=names)
    return {urn: names[0] for urn, names in normalized.items()}


def _build_mapping_dict(
    datasets: Sequence[PandasDataset],
    mappings: Optional[MappingsInput],
    input_names: List[str],
) -> Dict[str, List[str]]:
    """
    Build mapping dict from SDMX URNs to the VTL dataset names they feed.

    A single short-URN may map to several VTL dataset names, so the values are lists.

    Args:
        datasets: Sequence of PandasDataset objects.
        mappings: Optional mapping configuration (None, dict, VtlDataflowMapping,
            sequence of VtlDataflowMapping, or VtlMappingScheme).
        input_names: List of input dataset names extracted from the VTL script.

    Returns:
        Dict mapping short_urn -> list of vtl_dataset_names.

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
        return {schema.short_urn: [input_names[0]]}

    normalized = _normalize_mappings(mappings) or {}

    # A VTL dataset name may come from exactly one dataflow; duplicates across (or within)
    # dataflows would overwrite data silently.
    all_names = [name for names in normalized.values() for name in names]
    duplicates = sorted({name for name in all_names if all_names.count(name) > 1})
    if duplicates:
        raise InputValidationException(code="0-1-3-11", names=duplicates)

    return normalized


def _structure_dataset_names(structure: DataStructureItem) -> List[str]:
    """Return the dataset/scalar names declared in a VTL JSON structure dict.

    Only VTL JSON dicts expose their names cheaply; for other forms (Path, URL or
    pysdmx objects) an empty list is returned and no collision check is performed.
    """
    if not isinstance(structure, dict):
        return []
    names = [ds["name"] for ds in structure.get("datasets", []) if "name" in ds]
    names += [sc["name"] for sc in structure.get("scalars", []) if "name" in sc]
    return names


def _merge_sdmx_data_structures(
    sdmx_structures: List[Dict[str, Any]],
    extra: Optional[DataStructuresInput],
) -> List[DataStructureItem]:
    """
    Combine SDMX-derived VTL data structures with explicitly-provided ones.

    Args:
        sdmx_structures: VTL JSON structures derived from the SDMX datasets.
        extra: Additional data structures (same forms accepted by ``run``), or None.

    Returns:
        A single list with the SDMX-derived structures first, followed by the extras.

    Raises:
        InputValidationException: On duplicate dataset names between the SDMX-derived
            structures and dict-form extras (``0-1-3-9``).
    """
    combined: List[DataStructureItem] = list(sdmx_structures)
    if extra is None:
        return combined
    if isinstance(extra, list):
        combined.extend(extra)
    else:
        combined.append(extra)

    seen: List[str] = []
    for structure in combined:
        seen.extend(_structure_dataset_names(structure))
    duplicates = sorted({name for name in seen if seen.count(name) > 1})
    if duplicates:
        raise InputValidationException(code="0-1-3-9", names=duplicates)
    return combined


def _merge_sdmx_datapoints(
    sdmx_datapoints: Dict[str, Union[pd.DataFrame, str, Path]],
    extra: Optional[DatapointsInput],
) -> DatapointsInput:
    """
    Combine SDMX-derived datapoints with explicitly-provided ones.

    Args:
        sdmx_datapoints: Datapoints (as DataFrames) derived from the SDMX datasets.
        extra: Additional datapoints (same forms accepted by ``run``), or None.

    Returns:
        The merged datapoints. When ``extra`` is a dict it is merged by dataset name;
        otherwise it is only allowed when there are no SDMX-derived datapoints.

    Raises:
        InputValidationException: On duplicate dataset names (``0-1-3-9``), or when
            non-dict datapoints are combined with SDMX datasets (``0-1-3-10``).
    """
    if extra is None:
        return sdmx_datapoints
    if isinstance(extra, dict):
        duplicates = [name for name in extra if name in sdmx_datapoints]
        if duplicates:
            raise InputValidationException(code="0-1-3-9", names=duplicates)
        return {**sdmx_datapoints, **extra}
    if sdmx_datapoints:
        raise InputValidationException(code="0-1-3-10")
    return extra
