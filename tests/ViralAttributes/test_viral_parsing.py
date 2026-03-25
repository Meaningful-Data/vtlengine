import pandas as pd

from vtlengine import run
from vtlengine.Model import Role


class TestViralPropagationParsing:
    def test_parse_enumerated_propagation(self):
        """Enumerated viral propagation definition parses and runs without error."""
        script = """
            define viral propagation CONF_priority (variable At_1) is
                when "C" then "C";
                when "N" then "N";
                else "F"
            end viral propagation;

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
                },
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": ["A"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE

    def test_parse_aggregate_propagation(self):
        """Aggregate viral propagation definition parses and runs without error."""
        script = """
            define viral propagation TIME_prop (variable At_1) is
                aggr max
            end viral propagation;

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
                            "type": "Integer",
                            "role": "Viral Attribute",
                            "nullable": True,
                        },
                    ],
                },
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": [5]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components

    def test_parse_binary_clause_propagation(self):
        """Binary clause (when "A" and "B" then "C") parses correctly."""
        script = """
            define viral propagation COMP_mix (variable At_1) is
                when "C" and "M" then "N";
                when "M" then "M";
                else " "
            end viral propagation;

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
                },
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": ["C"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        assert "DS_r" in result

    def test_parse_valuedomain_propagation(self):
        """Value domain signature parses correctly."""
        script = """
            define viral propagation OBS_default (valuedomain CL_OBS_STATUS) is
                when "M" then "M";
                when "L" then "L";
                else "A"
            end viral propagation;

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
                    ],
                },
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        assert "DS_r" in result
