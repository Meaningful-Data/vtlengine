"""
Type aliases for the inputs of the public API and of the loaders behind it.

Each alias names one kind of input, so every function that takes it shares a single definition.
The aliases do not include ``None``: a parameter that can be left out is annotated as
``Optional[Alias] = None``.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd
from pysdmx.model import TransformationScheme
from pysdmx.model.dataflow import Dataflow, DataStructureDefinition, Schema
from pysdmx.model.vtl import VtlDataflowMapping

if TYPE_CHECKING:
    # Imported for type checking only: importing vtlengine.Model here, before the API has loaded
    # vtlengine.AST, starts the DataTypes <-> AST import cycle.
    from vtlengine.Model import ValueDomain

ScriptInput = Union[str, TransformationScheme, Path]
"""A VTL script as a string, a Path to a ``.vtl`` file or a pysdmx TransformationScheme."""

DataStructureInput = Union[str, Dict[str, Any], Path, Schema, DataStructureDefinition, Dataflow]
"""One data structure: VTL JSON (dict or Path), an SDMX file or URL, or a pysdmx object."""

DataStructuresInput = Union[DataStructureInput, List[DataStructureInput]]
"""One data structure or a list of them."""

DatapointsInput = Union[
    Dict[str, Union[pd.DataFrame, str, Path]], List[Union[str, Path]], str, Path
]
"""The datapoints: a dict of dataset name to DataFrame or file, or the files themselves."""

ValueDomainsInput = Union[
    Dict[str, Any], Path, "ValueDomain", List[Union[Dict[str, Any], Path, "ValueDomain"]]
]
"""Value domains as dicts, Paths to JSON files or folders, ValueDomain objects, or a list."""

ExternalRoutinesInput = Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]
"""External routines as dicts, Paths to JSON or SQL files or folders, or a list of them."""

ScalarValuesInput = Dict[str, Optional[Union[int, str, bool, float]]]
"""The values of the input scalars, by scalar name."""

SdmxMappingsInput = Union[VtlDataflowMapping, Dict[str, str]]
"""The mapping from SDMX dataset names or URNs to VTL dataset names."""
