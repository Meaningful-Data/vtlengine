import pandas as pd

from vtlengine import run
from vtlengine.Model import Role


class TestViralAttributeBinaryOps:
    def test_binary_propagation_single_viral_attribute(self):
        """Viral attribute from one operand is kept in result with same values."""
        script = "DS_r <- DS_1 + DS_2;"
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
                    ],
                },
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "At_1": ["A", "B"]}),
            "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 15.0]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(ds_r.data["At_1"]) == ["A", "B"]

    def test_binary_non_viral_attribute_still_dropped(self):
        """Non-viral (regular) attributes are still dropped as before."""
        script = "DS_r <- DS_1 + DS_2;"
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
                            "role": "Attribute",
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
                    ],
                },
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "At_1": ["A", "B"]}),
            "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 15.0]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" not in ds_r.components

    def test_binary_dataset_scalar_keeps_viral_attr(self):
        """DS_r <- DS_1 + 5 where DS_1 has a viral attribute."""
        script = "DS_r <- DS_1 + 5;"
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
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "At_1": ["A", "B"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(ds_r.data["At_1"]) == ["A", "B"]

    def test_binary_both_operands_viral_keeps_left(self):
        """When both operands have the same viral attr, keep left value."""
        script = "DS_r <- DS_1 + DS_2;"
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
            "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 15.0], "At_1": ["X", "Y"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        # Left operand values kept when no propagation rule
        assert list(ds_r.data["At_1"]) == ["A", "B"]
