from pathlib import Path

import pandas as pd
import pytest

from tests.Helper import TestHelper
from vtlengine import run
from vtlengine.API import create_ast
from vtlengine.DataTypes import Integer, Number, String
from vtlengine.Exceptions import VTLSyntaxError
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


execution_codes = [
    ("1-1", 2),
    ("1-2", 2),
    ("1-4", 2),
    ("1-5", 2),
    ("2-1", 1),
    ("2-2", 1),
    ("3-1", 1),
    ("7-1", 1),
    ("8-1", 1),
    ("10-3", 1),
    ("1-6", 2),
    ("1-7", 2),
    ("1-8", 2),
    ("2-3", 1),
    ("4-6", 2),
    ("4-7", 2),
    ("4-8", 2),
    ("4-10", 2),
    ("4-11", 2),
    ("4-12", 2),
    ("7-2", 1),
    ("7-3", 1),
    ("9-2", 1),
    ("11-1", 1),
    ("11-2", 1),
    ("11-3", 2),
    ("12-1", 1),
    ("12-2", 1),
    ("12-3", 2),
    ("12-4", 2),
    ("13-1", 1),
    ("13-2", 1),
]


@pytest.mark.parametrize("code,number_inputs", execution_codes)
def test_execution(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(code=code, number_inputs=number_inputs, references_names=["DS_r"])


# -- Combining a viral attribute without a rule is a semantic error (issue #877) --
combine_no_rule_codes = [
    ("1-3", 2, "1-3-3-6"),  # binary, both operands viral, no rule
    ("3-2", 1, "1-3-3-6"),  # analytic, viral combined over the partition, no rule
    ("4-9", 2, "1-3-3-6"),  # inner_join, viral in both operands, no rule
]


@pytest.mark.parametrize("code,number_inputs,exception_code", combine_no_rule_codes)
def test_combine_without_rule(code: str, number_inputs: int, exception_code: str) -> None:
    ViralHelper.NewSemanticExceptionTest(
        code=code, number_inputs=number_inputs, exception_code=exception_code
    )


semantic_codes = [
    ("5-1", 1),
    ("9-1", 2),
    ("10-1", 2),
    ("10-2", 1),
]


@pytest.mark.parametrize("code,number_inputs", semantic_codes)
def test_structure(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(
        code=code, number_inputs=number_inputs, references_names=["DS_r"], only_semantic=True
    )


statement_ordering_codes = [
    ("13-3", 1),
    ("13-4", 1),
]


@pytest.mark.parametrize("code,number_inputs", statement_ordering_codes)
def test_definition_statement_ordering(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(
        code=code, number_inputs=number_inputs, references_names=["DS_r"], only_semantic=True
    )


validation_codes = [
    ("6-1", "1-3-3-1"),
    ("6-2", "1-3-3-4"),
    ("6-3", "1-3-3-2"),
]


@pytest.mark.parametrize("code,exception_code", validation_codes)
def test_validation(code: str, exception_code: str) -> None:
    ViralHelper.NewSemanticExceptionTest(code=code, number_inputs=1, exception_code=exception_code)


join_ambiguity_codes = [
    ("4-1", "1-1-13-9"),
    ("4-2", "1-1-13-9"),
    ("4-3", "1-1-13-9"),
    ("4-4", "1-1-13-9"),
    ("4-5", "1-1-13-9"),
]


@pytest.mark.parametrize("code,exception_code", join_ambiguity_codes)
def test_join_ambiguity(code: str, exception_code: str) -> None:
    ViralHelper.NewSemanticExceptionTest(code=code, number_inputs=2, exception_code=exception_code)


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


class TestVpBodyGrammar:
    invalid_bodies = [
        pytest.param("aggregate max;\naggregate min", id="two_aggregates"),
        pytest.param('aggregate max;\nwhen "C" then "C"', id="aggregate_then_enumerated"),
        pytest.param('aggregate max;\nelse "F"', id="aggregate_then_else"),
        pytest.param('else "A";\nelse "B"', id="two_else"),
    ]

    @pytest.mark.parametrize("body", invalid_bodies)
    def test_invalid_vp_body_raises_syntax_error(self, body: str) -> None:
        script = f"define viral propagation R (variable VAt_1) is\n{body}\nend viral propagation;"
        with pytest.raises(VTLSyntaxError):
            create_ast(script)

    def test_valid_enumerated_then_else_propagates(self) -> None:
        """Valid body (enumerated clauses + trailing else): resolves on a binary op."""
        rule = (
            "define viral propagation R (variable VAt_1) is\n"
            'when "C" then "C";\nelse "F"\n'
            "end viral propagation;"
        )
        va = {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True},
            ],
        }
        result = run(
            script=rule + "\nDS_r <- DS_1 + DS_2;",
            data_structures={"datasets": [va, {**va, "name": "DS_2"}]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["C", "X"]}),
                "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 15.0], "VAt_1": ["C", "Y"]}),
            },
        )
        # C+C -> "C" (when "C"); X+Y -> "F" (else)
        assert list(result["DS_r"].data["VAt_1"]) == ["C", "F"]

    def test_valid_single_aggregate_propagates(self) -> None:
        """Valid body (single aggregate clause): max propagated through a group by."""
        rule = (
            "define viral propagation R (variable VAt_1) is\naggregate max\nend viral propagation;"
        )
        ds = {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Id_2", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "Integer", "role": "Viral Attribute", "nullable": True},
            ],
        }
        result = run(
            script=rule + "\nDS_r <- sum(DS_1 group by Id_1);",
            data_structures={"datasets": [ds]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {
                        "Id_1": [1, 1, 2],
                        "Id_2": [1, 2, 1],
                        "Me_1": [10.0, 20.0, 30.0],
                        "VAt_1": [3, 7, 5],
                    }
                )
            },
        )
        sorted_data = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        assert list(sorted_data["VAt_1"]) == [7, 5]
