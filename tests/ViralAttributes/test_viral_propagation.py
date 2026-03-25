"""Tests for define viral propagation: parsing, end-to-end execution, and semantic validation."""

import pandas as pd
import pytest

from vtlengine import run
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Role

# -- Shared fixtures --

SIMPLE_DS = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}
SIMPLE_DP = {"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0]})}


def _ds_with_viral(name: str, va_type: str = "String") -> dict:
    return {
        "name": name,
        "DataStructure": [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            {"name": "At_1", "type": va_type, "role": "Viral Attribute", "nullable": True},
        ],
    }


# -- Parsing tests --


class TestViralPropagationParsing:
    def test_parse_enumerated(self):
        script = """
            define viral propagation CONF (variable At_1) is
                when "C" then "C";
                when "N" then "N";
                else "F"
            end viral propagation;
            DS_r <- DS_1;
        """
        result = run(
            script=script,
            data_structures={"datasets": [_ds_with_viral("DS_1")]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": ["A"]})},
        )
        assert result["DS_r"].components["At_1"].role == Role.VIRAL_ATTRIBUTE

    def test_parse_aggregate(self):
        script = """
            define viral propagation T (variable At_1) is
                aggr max
            end viral propagation;
            DS_r <- DS_1;
        """
        result = run(
            script=script,
            data_structures={"datasets": [_ds_with_viral("DS_1", "Integer")]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": [5]})},
        )
        assert "At_1" in result["DS_r"].components

    def test_parse_binary_clause(self):
        script = """
            define viral propagation MIX (variable At_1) is
                when "C" and "M" then "N";
                when "M" then "M";
                else " "
            end viral propagation;
            DS_r <- DS_1;
        """
        result = run(
            script=script,
            data_structures={"datasets": [_ds_with_viral("DS_1")]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": ["C"]})},
        )
        assert "DS_r" in result

    def test_parse_valuedomain(self):
        script = """
            define viral propagation OBS (valuedomain CL_OBS) is
                when "M" then "M";
                else "A"
            end viral propagation;
            DS_r <- DS_1;
        """
        result = run(script=script, data_structures=SIMPLE_DS, datapoints=SIMPLE_DP)
        assert "DS_r" in result


# -- End-to-end propagation tests --


class TestViralPropagationEndToEnd:
    def test_enumerated_binary_propagation(self):
        """Unary clauses resolve viral attr values in binary operations."""
        script = """
            define viral propagation CONF (variable At_1) is
                when "C" then "C";
                when "N" then "N";
                else "F"
            end viral propagation;
            DS_r <- DS_1 + DS_2;
        """
        result = run(
            script=script,
            data_structures={"datasets": [_ds_with_viral("DS_1"), _ds_with_viral("DS_2")]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "At_1": ["C", "N", "F"]}
                ),
                "DS_2": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [5.0, 15.0, 25.0], "At_1": ["N", "F", "F"]}
                ),
            },
        )
        # C+N→C (unary "C"); N+F→N (unary "N"); F+F→F (else)
        assert list(result["DS_r"].data["At_1"]) == ["C", "N", "F"]

    def test_binary_clause_precedence(self):
        """Binary clauses take precedence over unary clauses."""
        script = """
            define viral propagation MIX (variable At_1) is
                when "C" and "M" then "N";
                when "M" then "M";
                else " "
            end viral propagation;
            DS_r <- DS_1 + DS_2;
        """
        result = run(
            script=script,
            data_structures={"datasets": [_ds_with_viral("DS_1"), _ds_with_viral("DS_2")]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "At_1": ["C", "M", "X"]}
                ),
                "DS_2": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [5.0, 15.0, 25.0], "At_1": ["M", "F", "Y"]}
                ),
            },
        )
        # C+M→N (binary); M+F→M (unary "M"); X+Y→" " (else)
        assert list(result["DS_r"].data["At_1"]) == ["N", "M", " "]

    def test_no_rule_gives_null(self):
        """Both operands viral but no rule defined → null."""
        result = run(
            script="DS_r <- DS_1 + DS_2;",
            data_structures={"datasets": [_ds_with_viral("DS_1"), _ds_with_viral("DS_2")]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": ["A"]}),
                "DS_2": pd.DataFrame({"Id_1": [1], "Me_1": [5.0], "At_1": ["B"]}),
            },
        )
        assert pd.isna(result["DS_r"].data["At_1"].iloc[0])

    def test_aggregate_max_in_aggregation(self):
        """Aggregate max propagation in group by."""
        script = """
            define viral propagation S (variable At_1) is
                aggr max
            end viral propagation;
            DS_r <- sum(DS_1 group by Id_1);
        """
        ds = {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Id_2", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "At_1", "type": "Integer", "role": "Viral Attribute", "nullable": True},
            ],
        }
        result = run(
            script=script,
            data_structures={"datasets": [ds]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {
                        "Id_1": [1, 1, 2],
                        "Id_2": [1, 2, 1],
                        "Me_1": [10.0, 20.0, 30.0],
                        "At_1": [3, 7, 5],
                    }
                )
            },
        )
        sorted_data = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        assert list(sorted_data["At_1"]) == [7, 5]


# -- Semantic validation tests --


class TestViralPropagationValidation:
    def test_duplicate_variable_rule_raises_error(self):
        script = """
            define viral propagation r1 (variable At_1) is
                when "C" then "C"
            end viral propagation;
            define viral propagation r2 (variable At_1) is
                when "N" then "N"
            end viral propagation;
            DS_r <- DS_1;
        """
        with pytest.raises(SemanticError, match="1-3-3-1"):
            run(script=script, data_structures=SIMPLE_DS, datapoints=SIMPLE_DP)

    def test_duplicate_enumeration_raises_error(self):
        script = """
            define viral propagation dup (variable At_1) is
                when "C" then "C";
                when "C" then "N"
            end viral propagation;
            DS_r <- DS_1;
        """
        with pytest.raises(SemanticError, match="1-3-3-4"):
            run(script=script, data_structures=SIMPLE_DS, datapoints=SIMPLE_DP)
