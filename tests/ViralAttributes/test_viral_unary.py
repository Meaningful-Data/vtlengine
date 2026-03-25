import pandas as pd
import pytest

from vtlengine import run
from vtlengine.Model import Role

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
    def test_numeric_unary_preserves_viral_attr(
        self, op: str, me_type: str, me_values: list
    ) -> None:
        script = f"DS_r <- {op}(DS_1);"
        data_structures = {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [
                        {
                            "name": "Id_1",
                            "type": "Integer",
                            "role": "Identifier",
                            "nullable": False,
                        },
                        {
                            "name": "Me_1",
                            "type": me_type,
                            "role": "Measure",
                            "nullable": True,
                        },
                        {
                            "name": "At_1",
                            "type": "String",
                            "role": "Viral Attribute",
                            "nullable": True,
                        },
                    ],
                }
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": me_values, "At_1": ["A", "B"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(ds_r.data["At_1"]) == ["A", "B"]

    @pytest.mark.parametrize(
        "op,me_type,me_values",
        STRING_UNARY_OPS,
        ids=[x[0] for x in STRING_UNARY_OPS],
    )
    def test_string_unary_preserves_viral_attr(
        self, op: str, me_type: str, me_values: list
    ) -> None:
        script = f"DS_r <- {op}(DS_1);"
        data_structures = {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [
                        {
                            "name": "Id_1",
                            "type": "Integer",
                            "role": "Identifier",
                            "nullable": False,
                        },
                        {
                            "name": "Me_1",
                            "type": me_type,
                            "role": "Measure",
                            "nullable": True,
                        },
                        {
                            "name": "At_1",
                            "type": "String",
                            "role": "Viral Attribute",
                            "nullable": True,
                        },
                    ],
                }
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": me_values, "At_1": ["A", "B"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(ds_r.data["At_1"]) == ["A", "B"]

    def test_not_preserves_viral_attr(self) -> None:
        script = "DS_r <- not DS_1;"
        data_structures = {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [
                        {
                            "name": "Id_1",
                            "type": "Integer",
                            "role": "Identifier",
                            "nullable": False,
                        },
                        {
                            "name": "Me_1",
                            "type": "Boolean",
                            "role": "Measure",
                            "nullable": True,
                        },
                        {
                            "name": "At_1",
                            "type": "String",
                            "role": "Viral Attribute",
                            "nullable": True,
                        },
                    ],
                }
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [True, False], "At_1": ["A", "B"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE

    def test_isnull_preserves_viral_attr(self) -> None:
        script = "DS_r <- isnull(DS_1);"
        data_structures = {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [
                        {
                            "name": "Id_1",
                            "type": "Integer",
                            "role": "Identifier",
                            "nullable": False,
                        },
                        {
                            "name": "Me_1",
                            "type": "Number",
                            "role": "Measure",
                            "nullable": True,
                        },
                        {
                            "name": "At_1",
                            "type": "String",
                            "role": "Viral Attribute",
                            "nullable": True,
                        },
                    ],
                }
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, None], "At_1": ["A", "B"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
