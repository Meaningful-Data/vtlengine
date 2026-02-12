"""
SDMX file handling utilities for VTL Engine.

This module consolidates all SDMX-related file operations including:
- Loading SDMX-ML (.xml) and SDMX-JSON (.json) datapoints
- Loading SDMX structure files
- Converting pysdmx objects to VTL JSON format
- Extracting dataset names from SDMX files
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import pandas as pd
from pysdmx.io import get_datasets as sdmx_get_datasets
from pysdmx.io.pd import PandasDataset
from pysdmx.model.dataflow import Component as SDMXComponent
from pysdmx.model.dataflow import Dataflow, DataStructureDefinition, Schema
from pysdmx.model.dataflow import Role as SDMX_Role

from vtlengine.Exceptions import DataLoadError, InputValidationException
from vtlengine.Model import Component, Role
from vtlengine.Utils import VTL_DTYPES_MAPPING, VTL_ROLE_MAPPING

# File extensions that trigger SDMX parsing when loading datapoints.
# .xml -> SDMX-ML (strict: raises error if parsing fails)
# .json -> SDMX-JSON (permissive: falls back to plain file if parsing fails)
SDMX_DATAPOINT_EXTENSIONS = {".xml", ".json"}

# File extensions that indicate SDMX structure files for data_structures parameter.
# .xml -> SDMX-ML structure (strict: raises error if parsing fails)
# .json -> SDMX-JSON structure (permissive: falls back to VTL JSON if parsing fails)
SDMX_STRUCTURE_EXTENSIONS = {".xml", ".json"}


def is_sdmx_datapoint_file(file_path: Path) -> bool:
    """Check if a file should be treated as SDMX when loading datapoints."""
    return file_path.suffix.lower() in SDMX_DATAPOINT_EXTENSIONS


def is_sdmx_structure_file(file_path: Path) -> bool:
    """Check if a file should be treated as SDMX structure file."""
    return file_path.suffix.lower() in SDMX_STRUCTURE_EXTENSIONS


def _extract_name_from_structure(
    structure: Union[str, Schema],
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> str:
    """
    Extract VTL dataset name from SDMX structure reference.

    Args:
        structure: Either a string URN or a Schema object from pysdmx.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.

    Returns:
        The VTL dataset name to use.
    """
    if isinstance(structure, str):
        # Check if mapping exists for this URN
        if sdmx_mappings and structure in sdmx_mappings:
            return sdmx_mappings[structure]
        # Extract short name from URN like "DataStructure=BIS:BIS_DER(1.0)" -> "BIS_DER"
        if "=" in structure and ":" in structure:
            parts = structure.split(":")
            return parts[-1].split("(")[0] if len(parts) >= 2 else structure
        return structure
    else:
        # Schema object - check mapping by short_urn first
        if (
            sdmx_mappings
            and hasattr(structure, "short_urn")
            and structure.short_urn in sdmx_mappings
        ):
            return sdmx_mappings[structure.short_urn]
        return structure.id


def extract_sdmx_dataset_name(
    file_path: Path,
    explicit_name: Optional[str] = None,
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> str:
    """
    Get the dataset name for an SDMX file by parsing its structure.

    Args:
        file_path: Path to the SDMX file.
        explicit_name: If provided, use this name directly.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.

    Returns:
        The dataset name to use.

    Raises:
        DataLoadError: If file cannot be parsed or contains no datasets.
    """
    if explicit_name is not None:
        return explicit_name

    try:
        pandas_datasets = cast(Sequence[PandasDataset], sdmx_get_datasets(data=file_path))
    except Exception as e:
        raise DataLoadError(
            code="0-3-1-8",
            file=str(file_path),
            error=str(e),
        )

    if not pandas_datasets:
        raise DataLoadError(
            code="0-3-1-9",
            file=str(file_path),
        )

    pd_dataset = pandas_datasets[0]
    return _extract_name_from_structure(pd_dataset.structure, sdmx_mappings)


def load_sdmx_datapoints(
    components: Dict[str, Component],
    dataset_name: str,
    file_path: Path,
) -> pd.DataFrame:
    """
    Load SDMX file (.xml or .json) and return DataFrame.

    Uses pysdmx to parse the file and extract data as a DataFrame.
    Handles SDMX-specific columns (DATAFLOW, STRUCTURE, ACTION, etc.)
    and validates that required identifiers are present.

    Args:
        components: Expected components for validation.
        dataset_name: Name of the dataset for error messages.
        file_path: Path to the SDMX file.

    Returns:
        pandas DataFrame with sanitized columns.

    Raises:
        DataLoadError: If file cannot be parsed or data is invalid.
        InputValidationException: If required identifiers are missing.
    """
    try:
        pandas_datasets = cast(Sequence[PandasDataset], sdmx_get_datasets(data=file_path))
    except Exception as e:
        raise DataLoadError(
            "0-3-1-8",
            file=str(file_path),
            error=str(e),
        )

    if not pandas_datasets:
        raise DataLoadError(
            "0-3-1-9",
            file=str(file_path),
        )

    # Use the first dataset
    pd_dataset: PandasDataset = pandas_datasets[0]
    data = pd_dataset.data

    # Sanitize SDMX-specific columns
    data = _sanitize_sdmx_columns(components, file_path, data)
    return data


def _sanitize_sdmx_columns(
    components: Dict[str, Component],
    file_path: Path,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove SDMX-specific columns and validate identifiers.

    Handles DATAFLOW, STRUCTURE, STRUCTURE_ID, and ACTION columns that
    are present in SDMX-CSV and SDMX-ML files but not part of VTL data.

    Args:
        components: Expected components for validation.
        file_path: Path to file for error messages.
        data: DataFrame to sanitize.

    Returns:
        Sanitized DataFrame.

    Raises:
        InputValidationException: If required identifiers are missing.
    """
    # Remove DATAFLOW column if present and not in components
    if (
        "DATAFLOW" in data.columns
        and data.columns[0] == "DATAFLOW"
        and "DATAFLOW" not in components
    ):
        data.drop(columns=["DATAFLOW"], inplace=True)

    # Remove STRUCTURE-related columns if present
    if "STRUCTURE" in data.columns and data.columns[0] == "STRUCTURE":
        if "STRUCTURE" not in components:
            data.drop(columns=["STRUCTURE"], inplace=True)
        if "STRUCTURE_ID" in data.columns:
            data.drop(columns=["STRUCTURE_ID"], inplace=True)
        # Handle ACTION column - remove deleted rows
        if "ACTION" in data.columns:
            data = data[data["ACTION"] != "D"]
            data.drop(columns=["ACTION"], inplace=True)

    # Validate identifiers are present
    comp_names = {c.name for c in components.values() if c.role == Role.IDENTIFIER}
    comps_missing = [id_m for id_m in comp_names if id_m not in data.columns]
    if comps_missing:
        comps_missing_str = ", ".join(comps_missing)
        raise InputValidationException(
            code="0-1-1-7", ids=comps_missing_str, file=str(file_path.name)
        )

    # Fill missing nullable components with None
    for comp_name, comp in components.items():
        if comp_name not in data:
            if not comp.nullable:
                raise InputValidationException(f"Component {comp_name} is missing in the file.")
            data[comp_name] = None

    return data


def load_sdmx_structure(
    file_path: Path,
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Load SDMX structure file and convert to VTL JSON format.

    Args:
        file_path: Path to SDMX structure file (.xml or .json).
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.

    Returns:
        VTL JSON data structure dict with 'datasets' key.

    Raises:
        DataLoadError: If file cannot be parsed or contains no structures.
    """
    from pysdmx.io import read_sdmx

    try:
        msg = read_sdmx(file_path)
    except Exception as e:
        raise DataLoadError(code="0-3-1-11", file=str(file_path), error=str(e))

    # Extract DataStructureDefinitions from the message
    structures = msg.structures if hasattr(msg, "structures") else None
    if structures is None or not structures:
        raise DataLoadError(code="0-3-1-12", file=str(file_path))

    # Filter to only include DataStructureDefinition objects
    dsds = [s for s in structures if isinstance(s, DataStructureDefinition)]
    if not dsds:
        raise DataLoadError(code="0-3-1-12", file=str(file_path))

    # Convert each DSD to VTL JSON and merge
    all_datasets: List[Dict[str, Any]] = []
    for dsd in dsds:
        # Determine dataset name: use mapping if available, otherwise use DSD ID
        dataset_name = dsd.id
        if sdmx_mappings and hasattr(dsd, "short_urn") and dsd.short_urn in sdmx_mappings:
            dataset_name = sdmx_mappings[dsd.short_urn]
        vtl_structure = to_vtl_json(dsd, dataset_name=dataset_name)
        all_datasets.extend(vtl_structure["datasets"])

    return {"datasets": all_datasets}


def to_vtl_json(
    structure: Union[DataStructureDefinition, Schema, Dataflow],
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert a pysdmx structure to VTL-compatible JSON representation.

    This function extracts and transforms the components (dimensions, measures,
    and attributes) from the given SDMX data structure and maps them into a
    dictionary format that conforms to the expected VTL data structure schema.

    Args:
        structure: An instance of DataStructureDefinition, Schema, or Dataflow.
        dataset_name: The name of the resulting VTL dataset. If not provided,
            uses the structure's ID (or Dataflow's ID for Dataflow objects).

    Returns:
        A dictionary representing the dataset in VTL format, with keys for
        dataset name and its components, including their name, role, data type,
        and nullability.

    Raises:
        InputValidationException: If a Dataflow has no associated DSD or if its
            structure is an unresolved reference.
    """
    # Handle Dataflow by extracting its DataStructureDefinition
    if isinstance(structure, Dataflow):
        if structure.structure is None:
            raise InputValidationException(
                f"Dataflow '{structure.id}' has no associated DataStructureDefinition."
            )
        if not isinstance(structure.structure, DataStructureDefinition):
            raise InputValidationException(
                f"Dataflow '{structure.id}' structure is a reference, not resolved. "
                "Please provide a resolved Dataflow with embedded DataStructureDefinition."
            )
        # Use Dataflow ID as dataset name if not provided
        if dataset_name is None:
            dataset_name = structure.id
        structure = structure.structure

    # Use structure ID if dataset_name not provided
    if dataset_name is None:
        dataset_name = structure.id

    components = []
    NAME = "name"
    ROLE = "role"
    TYPE = "type"
    NULLABLE = "nullable"

    _components: List[SDMXComponent] = []
    _components.extend(structure.components.dimensions)
    _components.extend(structure.components.measures)
    _components.extend(structure.components.attributes)

    for c in _components:
        _type = VTL_DTYPES_MAPPING[c.dtype]
        _nullability = c.role != SDMX_Role.DIMENSION
        _role = VTL_ROLE_MAPPING[c.role]

        component = {
            NAME: c.id,
            ROLE: _role,
            TYPE: _type,
            NULLABLE: _nullability,
        }

        components.append(component)

    result = {"datasets": [{"name": dataset_name, "DataStructure": components}]}

    return result
