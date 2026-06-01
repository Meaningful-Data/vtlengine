"""Viral attribute tests, unified in a single module.

File-based tests follow the repo convention: a VTL script + DataStructure JSON
(+ DataSet CSV) under ``data/``, addressed by a descriptive ``code`` and driven
through :class:`ViralHelper` (a :class:`tests.Helper.TestHelper` subclass):
- ``test_execution``   — full run, output structure AND data compared
- ``test_structure``   — semantic-only, output structure compared
- ``test_validation``  — expected semantic error code
The model-level checks (Role enum / Dataset helpers) live in ``TestViralAttributeRole``.
"""

from pathlib import Path

import pandas as pd
import pytest

from tests.Helper import TestHelper
from vtlengine.DataTypes import Integer, Number, String
from vtlengine.Model import Component, Dataset, Role, Role_keys


class ViralHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


# -- End-to-end execution: output structure AND data are compared --
# binary value resolution / aggregation / analytic / join merge, plus per-row
# passthrough (unary, dataset-scalar) and the legacy 'ViralAttribute' input role.
execution_codes = [
    ("binary_enumerated", 2),
    ("binary_clause_precedence", 2),
    ("binary_no_rule_null", 2),
    ("binary_one_operand", 2),
    ("binary_multi_attribute", 2),
    ("aggregation_max", 1),
    ("aggregation_enumerated", 1),
    ("analytic_max", 1),
    ("analytic_no_rule", 1),
    ("join_inner_merge", 2),
    ("join_no_rule_null", 2),
    ("join_left_max", 2),
    ("join_body_calc", 2),
    ("join_mixed_role", 2),
    ("unary_passthrough", 1),
    ("scalar_passthrough", 1),
    ("legacy_input", 1),
]


@pytest.mark.parametrize("code,number_inputs", execution_codes)
def test_execution(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(code=code, number_inputs=number_inputs, references_names=["DS_r"])


# -- Structure-only (semantic): the value-domain rule registers; viral attributes
# survive a set operator; non-viral attributes are dropped; calc creates a viral attr. --
semantic_codes = [
    ("parse_valuedomain", 1),
    ("intersect_preserved", 2),
    ("non_viral_dropped", 2),
    ("calc_viral", 1),
]


@pytest.mark.parametrize("code,number_inputs", semantic_codes)
def test_structure(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(
        code=code, number_inputs=number_inputs, references_names=["DS_r"], only_semantic=True
    )


# -- Semantic validation: duplicate variable rule / duplicate enumeration --
validation_codes = [
    ("validation_duplicate_variable", "1-3-3-1"),
    ("validation_duplicate_enumeration", "1-3-3-4"),
]


@pytest.mark.parametrize("code,exception_code", validation_codes)
def test_validation(code: str, exception_code: str) -> None:
    ViralHelper.NewSemanticExceptionTest(code=code, number_inputs=1, exception_code=exception_code)


# -- Model-level checks (Role enum and Dataset helpers) --
class TestViralAttributeRole:
    def test_viral_attribute_in_role_enum(self) -> None:
        assert Role.VIRAL_ATTRIBUTE.value == "Viral Attribute"

    def test_viral_attribute_in_role_keys(self) -> None:
        assert "Viral Attribute" in Role_keys

    def test_component_with_viral_attribute_role(self) -> None:
        comp = Component(name="VAt_1", data_type=String, role=Role.VIRAL_ATTRIBUTE, nullable=True)
        assert comp.role == Role.VIRAL_ATTRIBUTE

    def test_dataset_get_viral_attributes(self) -> None:
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component("Id_1", Integer, Role.IDENTIFIER, False),
                "Me_1": Component("Me_1", Number, Role.MEASURE, True),
                "VAt_1": Component("VAt_1", String, Role.VIRAL_ATTRIBUTE, True),
            },
            data=pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]}),
        )
        viral_attrs = ds.get_viral_attributes()
        assert len(viral_attrs) == 1
        assert viral_attrs[0].name == "VAt_1"

    def test_dataset_get_viral_attributes_names(self) -> None:
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component("Id_1", Integer, Role.IDENTIFIER, False),
                "VAt_1": Component("VAt_1", String, Role.VIRAL_ATTRIBUTE, True),
            },
            data=pd.DataFrame({"Id_1": [1], "VAt_1": ["A"]}),
        )
        assert ds.get_viral_attributes_names() == ["VAt_1"]

    def test_get_attributes_excludes_viral(self) -> None:
        """get_attributes() must NOT return VIRAL_ATTRIBUTE components."""
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component("Id_1", Integer, Role.IDENTIFIER, False),
                "VAt_1": Component("VAt_1", String, Role.ATTRIBUTE, True),
                "VAt_2": Component("VAt_2", String, Role.VIRAL_ATTRIBUTE, True),
            },
            data=pd.DataFrame({"Id_1": [1], "VAt_1": ["A"], "VAt_2": ["B"]}),
        )
        attrs = ds.get_attributes()
        assert len(attrs) == 1
        assert attrs[0].name == "VAt_1"
        assert "VAt_2" not in ds.get_attributes_names()
