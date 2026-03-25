import pandas as pd

from vtlengine import run
from vtlengine.Model import Role


class TestViralPropagationEndToEnd:
    def test_binary_with_enumerated_propagation(self):
        """End-to-end: define viral propagation + binary operator."""
        script = """
            define viral propagation CONF_priority (variable At_1) is
                when "C" then "C";
                when "N" then "N";
                else "F"
            end viral propagation;

            DS_r <- DS_1 + DS_2;
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
                {
                    "name": "DS_2",
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
            "DS_1": pd.DataFrame(
                {"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "At_1": ["C", "N", "F"]}
            ),
            "DS_2": pd.DataFrame(
                {"Id_1": [1, 2, 3], "Me_1": [5.0, 15.0, 25.0], "At_1": ["N", "F", "F"]}
            ),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
        # C+N → C (unary "C" matches); N+F → N (unary "N" matches); F+F → F (else)
        assert list(ds_r.data["At_1"]) == ["C", "N", "F"]

    def test_binary_with_binary_clause_precedence(self):
        """Binary clauses take precedence over unary clauses."""
        script = """
            define viral propagation COMP_mix (variable At_1) is
                when "C" and "M" then "N";
                when "M" then "M";
                else " "
            end viral propagation;

            DS_r <- DS_1 + DS_2;
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
                {
                    "name": "DS_2",
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
            "DS_1": pd.DataFrame(
                {"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "At_1": ["C", "M", "X"]}
            ),
            "DS_2": pd.DataFrame(
                {"Id_1": [1, 2, 3], "Me_1": [5.0, 15.0, 25.0], "At_1": ["M", "F", "Y"]}
            ),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        # C+M → N (binary clause); M+F → M (unary "M"); X+Y → " " (else/default)
        assert list(ds_r.data["At_1"]) == ["N", "M", " "]

    def test_no_rule_gives_null(self):
        """When both operands have same viral attr but no propagation rule, result is null."""
        script = """
            DS_r <- DS_1 + DS_2;
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
                {
                    "name": "DS_2",
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
            "DS_2": pd.DataFrame({"Id_1": [1], "Me_1": [5.0], "At_1": ["B"]}),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        # No propagation rule → null
        assert pd.isna(ds_r.data["At_1"].iloc[0])

    def test_aggregate_propagation_with_max(self):
        """End-to-end: aggregate max propagation in aggregation."""
        script = """
            define viral propagation SCORE_prop (variable At_1) is
                aggr max
            end viral propagation;

            DS_r <- sum(DS_1 group by Id_1);
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
                        {
                            "name": "Id_2",
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
                }
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame(
                {
                    "Id_1": [1, 1, 2],
                    "Id_2": [1, 2, 1],
                    "Me_1": [10.0, 20.0, 30.0],
                    "At_1": [3, 7, 5],
                }
            ),
        }
        result = run(script=script, data_structures=data_structures, datapoints=datapoints)
        ds_r = result["DS_r"]
        assert "At_1" in ds_r.components
        assert ds_r.components["At_1"].role == Role.VIRAL_ATTRIBUTE
        # Group Id_1=1: max(3,7)=7; Group Id_1=2: max(5)=5
        sorted_data = ds_r.data.sort_values("Id_1").reset_index(drop=True)
        assert list(sorted_data["At_1"]) == [7, 5]
