import pandas as pd

from vtlengine import run
from vtlengine.Model import Role


class TestViralAttributeAggregation:
    def test_aggregation_keeps_viral_attribute(self):
        script = "DS_r <- sum(DS_1 group by Id_1);"
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
                            "name": "Id_2",
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
            "DS_1": pd.DataFrame(
                {
                    "Id_1": [1, 1, 2],
                    "Id_2": [1, 2, 1],
                    "Me_1": [10.0, 20.0, 30.0],
                    "At_1": ["A", "A", "B"],
                }
            ),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE


class TestViralAttributeBetween:
    def test_between_keeps_viral_attribute(self):
        script = "DS_r <- between(DS_1, 5, 25);"
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
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 30.0], "At_1": ["A", "B"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE


class TestViralAttributeSetOps:
    def test_intersect_preserves_viral_attribute(self):
        script = "DS_r <- intersect(DS_1, DS_2);"
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
                },
                {
                    "name": "DS_2",
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
                },
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "At_1": ["A", "B"]}),
            "DS_2": pd.DataFrame({"Id_1": [1, 3], "Me_1": [10.0, 30.0], "At_1": ["A", "C"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
