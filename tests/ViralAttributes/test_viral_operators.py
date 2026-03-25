"""Tests for viral attribute propagation through all operator categories."""

import pandas as pd
import pytest

from vtlengine import run
from vtlengine.Model import Role

# -- Shared data structure builders --


def _ds(name: str, components: list) -> dict:
    return {"name": name, "DataStructure": components}


def _id(name: str = "Id_1") -> dict:
    return {"name": name, "type": "Integer", "role": "Identifier", "nullable": False}


def _me(name: str = "Me_1", type_: str = "Number") -> dict:
    return {"name": name, "type": type_, "role": "Measure", "nullable": True}


def _va(name: str = "At_1", type_: str = "String") -> dict:
    return {"name": name, "type": type_, "role": "Viral Attribute", "nullable": True}


def _at(name: str = "At_1", type_: str = "String") -> dict:
    return {"name": name, "type": type_, "role": "Attribute", "nullable": True}


def _run(script: str, datasets: list, datapoints: dict) -> dict:
    return run(
        script=script,
        data_structures={"datasets": datasets},
        datapoints=datapoints,
    )


# -- Calc / Input loading --


class TestViralAttributeInputAndCalc:
    def test_calc_viral_attribute(self):
        result = _run(
            'DS_r <- DS_1 [calc viral attribute At_1 := "X"];',
            [_ds("DS_1", [_id(), _me()])],
            {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0]})},
        )
        assert result["DS_r"].components["At_1"].role == Role.VIRAL_ATTRIBUTE

    def test_input_viral_attribute_loaded(self):
        result = _run(
            "DS_r <- DS_1;",
            [_ds("DS_1", [_id(), _me(), _va()])],
            {"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": ["A"]})},
        )
        assert result["DS_r"].components["At_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(result["DS_r"].data["At_1"]) == ["A"]

    def test_input_viral_attribute_legacy_format(self):
        ds = {
            "name": "DS_1",
            "DataStructure": [
                _id(),
                _me(),
                {"name": "At_1", "type": "String", "role": "ViralAttribute", "nullable": True},
            ],
        }
        result = _run(
            "DS_r <- DS_1;",
            [ds],
            {"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": ["A"]})},
        )
        assert result["DS_r"].components["At_1"].role == Role.VIRAL_ATTRIBUTE


# -- Binary operators --


class TestViralAttributeBinaryOps:
    def test_single_viral_attribute_propagated(self):
        result = _run(
            "DS_r <- DS_1 + DS_2;",
            [_ds("DS_1", [_id(), _me(), _va()]), _ds("DS_2", [_id(), _me()])],
            {
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "At_1": ["A", "B"]}),
                "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 15.0]}),
            },
        )
        ds_r = result["DS_r"]
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(ds_r.data["At_1"]) == ["A", "B"]

    def test_non_viral_attribute_still_dropped(self):
        result = _run(
            "DS_r <- DS_1 + DS_2;",
            [_ds("DS_1", [_id(), _me(), _at()]), _ds("DS_2", [_id(), _me()])],
            {
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "At_1": ["A", "B"]}),
                "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 15.0]}),
            },
        )
        assert "At_1" not in result["DS_r"].components

    def test_dataset_scalar_keeps_viral_attr(self):
        result = _run(
            "DS_r <- DS_1 + 5;",
            [_ds("DS_1", [_id(), _me(), _va()])],
            {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "At_1": ["A", "B"]})},
        )
        ds_r = result["DS_r"]
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(ds_r.data["At_1"]) == ["A", "B"]

    def test_both_operands_viral_no_rule_gives_null(self):
        result = _run(
            "DS_r <- DS_1 + DS_2;",
            [_ds("DS_1", [_id(), _me(), _va()]), _ds("DS_2", [_id(), _me(), _va()])],
            {
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": ["A"]}),
                "DS_2": pd.DataFrame({"Id_1": [1], "Me_1": [5.0], "At_1": ["B"]}),
            },
        )
        assert all(pd.isna(v) for v in result["DS_r"].data["At_1"])


# -- Unary operators --

NUMERIC_UNARY_OPS = [
    ("abs", "Number", [10.0, -20.0]),
    ("ceil", "Number", [1.1, 2.9]),
    ("floor", "Number", [1.1, 2.9]),
    ("ln", "Number", [1.0, 2.718]),
    ("exp", "Number", [0.0, 1.0]),
    ("sqrt", "Number", [4.0, 9.0]),
]

STRING_UNARY_OPS = [
    ("upper", "String", ["abc", "def"]),
    ("lower", "String", ["ABC", "DEF"]),
    ("trim", "String", [" abc ", " def "]),
    ("ltrim", "String", [" abc", " def"]),
    ("rtrim", "String", ["abc ", "def "]),
    ("length", "String", ["abc", "defgh"]),
]


class TestViralAttributeUnaryOps:
    @pytest.mark.parametrize(
        "op,me_type,me_values",
        NUMERIC_UNARY_OPS,
        ids=[x[0] for x in NUMERIC_UNARY_OPS],
    )
    def test_numeric_unary_preserves_viral_attr(self, op, me_type, me_values):
        result = _run(
            f"DS_r <- {op}(DS_1);",
            [_ds("DS_1", [_id(), _me(type_=me_type), _va()])],
            {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": me_values, "At_1": ["A", "B"]})},
        )
        assert result["DS_r"].components["At_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(result["DS_r"].data["At_1"]) == ["A", "B"]

    @pytest.mark.parametrize(
        "op,me_type,me_values",
        STRING_UNARY_OPS,
        ids=[x[0] for x in STRING_UNARY_OPS],
    )
    def test_string_unary_preserves_viral_attr(self, op, me_type, me_values):
        result = _run(
            f"DS_r <- {op}(DS_1);",
            [_ds("DS_1", [_id(), _me(type_=me_type), _va()])],
            {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": me_values, "At_1": ["A", "B"]})},
        )
        assert result["DS_r"].components["At_1"].role == Role.VIRAL_ATTRIBUTE

    def test_not_preserves_viral_attr(self):
        result = _run(
            "DS_r <- not DS_1;",
            [_ds("DS_1", [_id(), _me(type_="Boolean"), _va()])],
            {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [True, False], "At_1": ["A", "B"]})},
        )
        assert result["DS_r"].components["At_1"].role == Role.VIRAL_ATTRIBUTE

    def test_isnull_preserves_viral_attr(self):
        result = _run(
            "DS_r <- isnull(DS_1);",
            [_ds("DS_1", [_id(), _me(), _va()])],
            {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, None], "At_1": ["A", "B"]})},
        )
        assert result["DS_r"].components["At_1"].role == Role.VIRAL_ATTRIBUTE


# -- Other operators --


class TestViralAttributeOtherOps:
    def test_aggregation_keeps_viral_attribute(self):
        result = _run(
            "DS_r <- sum(DS_1 group by Id_1);",
            [_ds("DS_1", [_id(), _id("Id_2"), _me(), _va()])],
            {
                "DS_1": pd.DataFrame(
                    {
                        "Id_1": [1, 1, 2],
                        "Id_2": [1, 2, 1],
                        "Me_1": [10.0, 20.0, 30.0],
                        "At_1": ["A", "A", "B"],
                    }
                )
            },
        )
        assert result["DS_r"].components["At_1"].role == Role.VIRAL_ATTRIBUTE

    def test_between_keeps_viral_attribute(self):
        result = _run(
            "DS_r <- between(DS_1, 5, 25);",
            [_ds("DS_1", [_id(), _me(), _va()])],
            {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 30.0], "At_1": ["A", "B"]})},
        )
        assert result["DS_r"].components["At_1"].role == Role.VIRAL_ATTRIBUTE

    def test_intersect_preserves_viral_attribute(self):
        result = _run(
            "DS_r <- intersect(DS_1, DS_2);",
            [_ds("DS_1", [_id(), _me(), _va()]), _ds("DS_2", [_id(), _me(), _va()])],
            {
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "At_1": ["A", "B"]}),
                "DS_2": pd.DataFrame({"Id_1": [1, 3], "Me_1": [10.0, 30.0], "At_1": ["A", "C"]}),
            },
        )
        assert result["DS_r"].components["At_1"].role == Role.VIRAL_ATTRIBUTE
