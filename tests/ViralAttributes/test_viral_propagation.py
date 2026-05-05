"""Tests for define viral propagation: parsing, end-to-end execution, and semantic validation."""

import pandas as pd
import pytest

from vtlengine import run
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Role

# -- Shared propagation rules --

CONF_RULE = """
    define viral propagation CONF (variable VAt_1) is
        when "C" then "C";
        when "N" then "N";
        else "F"
    end viral propagation;
"""

CONF_BINARY_RULE = """
    define viral propagation COMP_mix (variable VAt_1) is
        when "C" and "M" then "N";
        when "M" then "M";
        else " "
    end viral propagation;
"""

TWO_RULES = """
    define viral propagation R1 (variable VAt_1) is
        when "C" then "C";
        when "N" then "N";
        else "F"
    end viral propagation;
    define viral propagation R2 (variable VAt_2) is
        aggr max
    end viral propagation;
"""

AGGR_MAX_RULE = """
    define viral propagation S (variable VAt_1) is
        aggr max
    end viral propagation;
"""

# -- Shared datasets --

DS_1VA = {
    "name": "DS_1",
    "DataStructure": [
        {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
        {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
        {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True},
    ],
}

DS_2VA = {
    "name": "DS_1",
    "DataStructure": [
        {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
        {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
        {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True},
        {"name": "VAt_2", "type": "Integer", "role": "Viral Attribute", "nullable": True},
    ],
}

DS_NO_VA = {
    "name": "DS_1",
    "DataStructure": [
        {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
        {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
    ],
}

SIMPLE_DS = {"datasets": [DS_NO_VA]}
SIMPLE_DP = {"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0]})}


def _ds_pair(ds_def: dict) -> dict:
    """Create a two-dataset structure from a single definition (DS_1, DS_2)."""
    ds2 = {**ds_def, "name": "DS_2"}
    return {"datasets": [ds_def, ds2]}


# -- Parsing tests --


class TestViralPropagationParsing:
    def test_parse_enumerated(self) -> None:
        result = run(
            script=CONF_RULE + "DS_r <- DS_1;",
            data_structures={"datasets": [DS_1VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]})},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE

    def test_parse_aggregate(self) -> None:
        ds = {
            **DS_NO_VA,
            "DataStructure": [
                *DS_NO_VA["DataStructure"],
                {"name": "VAt_1", "type": "Integer", "role": "Viral Attribute", "nullable": True},
            ],
        }
        result = run(
            script=AGGR_MAX_RULE + "DS_r <- DS_1;",
            data_structures={"datasets": [ds]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": [5]})},
        )
        assert "VAt_1" in result["DS_r"].components

    def test_parse_binary_clause(self) -> None:
        result = run(
            script=CONF_BINARY_RULE + "DS_r <- DS_1;",
            data_structures={"datasets": [DS_1VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["C"]})},
        )
        assert "DS_r" in result

    def test_parse_valuedomain(self) -> None:
        script = """
            define viral propagation OBS (valuedomain CL_OBS) is
                when "M" then "M";
                else "A"
            end viral propagation;
            DS_r <- DS_1;
        """
        result = run(script=script, data_structures=SIMPLE_DS, datapoints=SIMPLE_DP)
        assert "DS_r" in result


# -- End-to-end propagation (single viral attribute) --

propagation_binary_params = [
    "DS_1 + DS_2",
    "DS_1 - DS_2",
    "DS_1 * DS_2",
]


class TestViralPropagationEndToEnd:
    @pytest.mark.parametrize("expr", propagation_binary_params)
    def test_enumerated_propagation_binary(self, expr: str) -> None:
        """Same CONF_RULE resolution regardless of binary operator."""
        result = run(
            script=CONF_RULE + f"DS_r <- {expr};",
            data_structures=_ds_pair(DS_1VA),
            datapoints={
                "DS_1": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "VAt_1": ["C", "N", "F"]}
                ),
                "DS_2": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [5.0, 15.0, 25.0], "VAt_1": ["N", "F", "F"]}
                ),
            },
        )
        # C+N→C (unary "C"); N+F→N (unary "N"); F+F→F (else)
        assert list(result["DS_r"].data["VAt_1"]) == ["C", "N", "F"]

    def test_binary_clause_precedence(self) -> None:
        """Binary clauses take precedence over unary clauses."""
        result = run(
            script=CONF_BINARY_RULE + "DS_r <- DS_1 + DS_2;",
            data_structures=_ds_pair(DS_1VA),
            datapoints={
                "DS_1": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "VAt_1": ["C", "M", "X"]}
                ),
                "DS_2": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [5.0, 15.0, 25.0], "VAt_1": ["M", "F", "Y"]}
                ),
            },
        )
        # C+M→N (binary); M+F→M (unary "M"); X+Y→" " (else)
        assert list(result["DS_r"].data["VAt_1"]) == ["N", "M", " "]

    def test_no_rule_gives_null(self) -> None:
        """Both operands viral but no rule defined → null."""
        result = run(
            script="DS_r <- DS_1 + DS_2;",
            data_structures=_ds_pair(DS_1VA),
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]}),
                "DS_2": pd.DataFrame({"Id_1": [1], "Me_1": [5.0], "VAt_1": ["B"]}),
            },
        )
        assert pd.isna(result["DS_r"].data["VAt_1"].iloc[0])

    def test_aggregate_max_in_aggregation(self) -> None:
        """Aggregate max propagation in group by."""
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
            script=AGGR_MAX_RULE + "DS_r <- sum(DS_1 group by Id_1);",
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


# -- Multi-attribute propagation (enumerated + aggregate in one script) --


class TestViralPropagationMultiAttribute:
    @pytest.mark.parametrize("expr", propagation_binary_params)
    def test_two_rules_two_attrs_binary(self, expr: str) -> None:
        """TWO_RULES: VAt_1 enumerated + VAt_2 aggr max, applied to binary ops."""
        result = run(
            script=TWO_RULES + f"DS_r <- {expr};",
            data_structures=_ds_pair(DS_2VA),
            datapoints={
                "DS_1": pd.DataFrame(
                    {
                        "Id_1": [1, 2, 3],
                        "Me_1": [10.0, 20.0, 30.0],
                        "VAt_1": ["C", "N", "F"],
                        "VAt_2": [3, 5, 1],
                    }
                ),
                "DS_2": pd.DataFrame(
                    {
                        "Id_1": [1, 2, 3],
                        "Me_1": [5.0, 15.0, 25.0],
                        "VAt_1": ["N", "F", "F"],
                        "VAt_2": [7, 2, 4],
                    }
                ),
            },
        )
        ds_r = result["DS_r"]
        # VAt_1 enumerated: C+N→C, N+F→N, F+F→F
        assert list(ds_r.data["VAt_1"]) == ["C", "N", "F"]
        # VAt_2 aggr max: max(3,7)=7, max(5,2)=5, max(1,4)=4
        assert list(ds_r.data["VAt_2"]) == [7, 5, 4]


# -- Semantic validation --


class TestViralPropagationValidation:
    def test_duplicate_variable_rule_raises_error(self) -> None:
        script = """
            define viral propagation r1 (variable VAt_1) is
                when "C" then "C"
            end viral propagation;
            define viral propagation r2 (variable VAt_1) is
                when "N" then "N"
            end viral propagation;
            DS_r <- DS_1;
        """
        with pytest.raises(SemanticError, match="1-3-3-1"):
            run(script=script, data_structures=SIMPLE_DS, datapoints=SIMPLE_DP)

    def test_duplicate_enumeration_raises_error(self) -> None:
        script = """
            define viral propagation dup (variable VAt_1) is
                when "C" then "C";
                when "C" then "N"
            end viral propagation;
            DS_r <- DS_1;
        """
        with pytest.raises(SemanticError, match="1-3-3-4"):
            run(script=script, data_structures=SIMPLE_DS, datapoints=SIMPLE_DP)
