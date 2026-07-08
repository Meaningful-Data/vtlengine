"""Viral attribute tests, unified in a single module.

File-based tests follow the repo convention: a VTL script + DataStructure JSON
(+ DataSet CSV) under ``data/``, addressed by a numeric ``code`` and driven
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
execution_codes = [
    ("1-1", 2),  # binary, enumerated rule
    ("1-2", 2),  # binary, clause precedence (binary clause before unary)
    ("1-4", 2),  # binary, viral attr in one operand only (passthrough)
    ("1-5", 2),  # binary, two rules (enumerated VAt_1 + aggregate-max VAt_2)
    ("2-1", 1),  # aggregation, aggregate-max
    ("2-2", 1),  # aggregation, enumerated pairwise reduction
    ("3-1", 1),  # analytic, aggregate-max over the partition
    ("7-1", 1),  # unary (abs), per-row passthrough
    ("8-1", 1),  # dataset-scalar (DS_1 + 5), per-row passthrough
    ("10-3", 1),  # legacy 'ViralAttribute' input role
]


@pytest.mark.parametrize("code,number_inputs", execution_codes)
def test_execution(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(code=code, number_inputs=number_inputs, references_names=["DS_r"])


# -- Combining a viral attribute without a rule is a semantic error (issue #877) --
combine_no_rule_codes = [
    ("1-3", 2, "1-3-3-6"),  # binary, both operands viral, no rule
    ("3-2", 1, "1-3-3-6"),  # analytic, viral combined over the partition, no rule
]


@pytest.mark.parametrize("code,number_inputs,exception_code", combine_no_rule_codes)
def test_combine_without_rule(code: str, number_inputs: int, exception_code: str) -> None:
    ViralHelper.NewSemanticExceptionTest(
        code=code, number_inputs=number_inputs, exception_code=exception_code
    )


# -- Structure-only (semantic) --
semantic_codes = [
    ("5-1", 1),  # a value-domain rule registers; viral attr preserved
    ("9-1", 2),  # intersect preserves viral attributes
    ("10-1", 2),  # a non-viral attribute is still dropped
    ("10-2", 1),  # calc creates a viral attribute
]


@pytest.mark.parametrize("code,number_inputs", semantic_codes)
def test_structure(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(
        code=code, number_inputs=number_inputs, references_names=["DS_r"], only_semantic=True
    )


# -- Semantic validation: duplicate rules --
validation_codes = [
    ("6-1", "1-3-3-1"),  # duplicate variable-level rule
    ("6-2", "1-3-3-4"),  # duplicate enumeration combination
    ("6-3", "1-3-3-2"),  # duplicate value-domain-level rule
]


@pytest.mark.parametrize("code,exception_code", validation_codes)
def test_validation(code: str, exception_code: str) -> None:
    ViralHelper.NewSemanticExceptionTest(code=code, number_inputs=1, exception_code=exception_code)


# -- Join ambiguity: a component shared (and not disambiguated) by both join
# operands collapses to a homonym at the join's final un-prefixing step, which
# VTL 2.2 rejects. Valid viral-propagation-through-join cases (distinct
# measures) are covered in test_viral_propagation.TestViralPropagationJoins.
join_ambiguity_codes = [
    ("4-1", "1-1-13-9"),  # inner join, shared measure Me_1
    ("4-2", "1-1-13-9"),  # inner join, shared measure Me_1
    ("4-3", "1-1-13-9"),  # left join, shared measure Me_1
    ("4-4", "1-1-13-9"),  # join with body calc, shared measure Me_1 still present
    ("4-5", "1-1-13-9"),  # join, mixed-role shared component VAt_1
]


@pytest.mark.parametrize("code,exception_code", join_ambiguity_codes)
def test_join_ambiguity(code: str, exception_code: str) -> None:
    ViralHelper.NewSemanticExceptionTest(code=code, number_inputs=2, exception_code=exception_code)


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
