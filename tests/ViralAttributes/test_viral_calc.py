import pandas as pd

from vtlengine import run
from vtlengine.Model import Role


class TestViralAttributeCalc:
    def test_calc_viral_attribute(self):
        """Test that 'viral attribute' role setter works in calc clause."""
        script = """
            DS_r <- DS_1 [calc viral attribute At_1 := "X"];
        """
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
                        {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                    ],
                }
            ]
        }
        datapoints = {"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0]})}
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(ds_r.data["At_1"]) == ["X", "X"]

    def test_input_viral_attribute_loaded(self):
        """Test that a viral attribute in input data structure is correctly loaded."""
        script = """
            DS_r <- DS_1;
        """
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
                        {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
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
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "At_1": ["A", "B"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(ds_r.data["At_1"]) == ["A", "B"]

    def test_input_viral_attribute_legacy_format(self):
        """Test that 'ViralAttribute' (legacy format) is correctly loaded."""
        script = """
            DS_r <- DS_1;
        """
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
                        {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                        {
                            "name": "At_1",
                            "type": "String",
                            "role": "ViralAttribute",
                            "nullable": True,
                        },
                    ],
                }
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "At_1": ["A", "B"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
