"""Tests for define viral propagation: parsing, end-to-end execution, and semantic validation."""

from typing import Optional

import pandas as pd
import pytest

from vtlengine import run
from vtlengine.API import create_ast
from vtlengine.Exceptions import SemanticError, VTLSyntaxError
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
        aggregate max
    end viral propagation;
"""

AGGR_MAX_RULE = """
    define viral propagation S (variable VAt_1) is
        aggregate max
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

    @pytest.mark.parametrize("use_duckdb", [False, True])
    def test_null_condition_matches_null_value(self, use_duckdb: bool) -> None:
        """A `when null` clause must match a null (pd.NA/None) viral value, not fall to else."""
        result = run(
            script=(
                "define viral propagation ee (variable VAt_1) is\n"
                '    when null then "Nullable";\n'
                '    else "NO_COINCIDENCE"\n'
                "end viral propagation;\n"
                "DS_r <- DS_1 + DS_2;"
            ),
            data_structures=_ds_pair(DS_1VA),
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["X", None]}),
                "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 15.0], "VAt_1": ["X", None]}),
            },
            use_duckdb=use_duckdb,
        )
        # X+X→else "NO_COINCIDENCE"; null+null→"Nullable" (when null) on both engines.
        assert list(result["DS_r"].data["VAt_1"]) == ["NO_COINCIDENCE", "Nullable"]

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

    @pytest.mark.parametrize(
        "agg, both_present, one_null",
        # min/max skip nulls (LEAST/GREATEST); sum/avg propagate nulls (a + b).
        [("min", 10, 5), ("max", 20, 5), ("sum", 30, None), ("avg", 15, None)],
    )
    @pytest.mark.parametrize("use_duckdb", [False, True])
    def test_aggregate_binary_with_nulls(
        self, agg: str, both_present: int, one_null: Optional[int], use_duckdb: bool
    ) -> None:
        """Aggregate viral propagation on a binary op handles nulls identically on both engines.

        min/max ignore a null operand (SQL LEAST/GREATEST); sum/avg propagate it (a + b).
        Two nulls always yield null. Pandas and DuckDB must agree.
        """
        num_va_ds = {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "Number", "role": "Viral Attribute", "nullable": True},
            ],
        }
        rule = (
            f"define viral propagation ee (variable VAt_1) is\n"
            f"    aggregate {agg}\n"
            f"end viral propagation;\n"
        )
        result = run(
            script=rule + "DS_r <- DS_1 + DS_2;",
            data_structures=_ds_pair(num_va_ds),
            datapoints={
                "DS_1": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "VAt_1": [10.0, None, None]}
                ),
                "DS_2": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [5.0, 15.0, 25.0], "VAt_1": [20.0, 5.0, None]}
                ),
            },
            use_duckdb=use_duckdb,
        )
        va = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)["VAt_1"]
        # row1: both present; row2: one operand null; row3: both null -> always null
        assert int(va.iloc[0]) == both_present
        if one_null is None:
            assert pd.isna(va.iloc[1])
        else:
            assert int(va.iloc[1]) == one_null
        assert pd.isna(va.iloc[2])


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


# -- vpBody grammar restriction (VTL 2.2 reference manual) --


def _vp(body: str) -> str:
    return f"define viral propagation R (variable VAt_1) is\n{body}\nend viral propagation;"


class TestVpBodyGrammar:
    # Bodies that violate the spec and must be rejected at parse time.
    invalid_bodies = [
        pytest.param("aggregate max;\naggregate min", id="two_aggregates"),
        pytest.param('aggregate max;\nwhen "C" then "C"', id="aggregate_then_enumerated"),
        pytest.param('aggregate max;\nelse "F"', id="aggregate_then_else"),
        pytest.param('else "A";\nelse "B"', id="two_else"),
    ]

    @pytest.mark.parametrize("body", invalid_bodies)
    def test_invalid_vp_body_raises_syntax_error(self, body: str) -> None:
        with pytest.raises(VTLSyntaxError):
            create_ast(_vp(body))

    def test_valid_enumerated_then_else_propagates(self) -> None:
        """Valid body (enumerated clauses + trailing else): resolves on a binary op."""
        script = _vp('when "C" then "C";\nelse "F"') + "\nDS_r <- DS_1 + DS_2;"
        result = run(
            script=script,
            data_structures=_ds_pair(DS_1VA),
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["C", "X"]}),
                "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 15.0], "VAt_1": ["C", "Y"]}),
            },
        )
        # C+C→"C" (when "C"); X+Y→"F" (else)
        assert list(result["DS_r"].data["VAt_1"]) == ["C", "F"]

    def test_valid_single_aggregate_propagates(self) -> None:
        """Valid body (single aggregate clause): max propagated through a group by."""
        ds = {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Id_2", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "Integer", "role": "Viral Attribute", "nullable": True},
            ],
        }
        script = _vp("aggregate max") + "\nDS_r <- sum(DS_1 group by Id_1);"
        result = run(
            script=script,
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


# -- Propagation rules through join operators --

_ID_1 = {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False}
_ID_2 = {"name": "Id_2", "type": "Integer", "role": "Identifier", "nullable": False}
_ME_1 = {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True}
_ME_2 = {"name": "Me_2", "type": "Number", "role": "Measure", "nullable": True}
_VA = {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True}

DS_JOIN_1 = {"name": "DS_1", "DataStructure": [_ID_1, _ME_1, _VA]}
DS_JOIN_2 = {"name": "DS_2", "DataStructure": [_ID_1, _ME_2, _VA]}
DS_JOIN_2_NO_VA = {"name": "DS_2", "DataStructure": [_ID_1, _ME_2]}
DS_CROSS_2 = {"name": "DS_2", "DataStructure": [_ID_2, _ME_2, _VA]}
DS_CROSS_2_NO_VA = {"name": "DS_2", "DataStructure": [_ID_2, _ME_2]}


class TestViralPropagationJoins:
    """A viral attribute shared by both join operands is combined with the
    Attribute Propagation Rule, exactly like in binary operators; a viral
    attribute coming from a single operand is carried over unchanged. This holds
    for all four join operators, ``cross_join`` included."""

    @pytest.mark.parametrize("join_op", ["inner_join", "left_join", "full_join"])
    def test_enumerated_propagation_join(self, join_op: str) -> None:
        """Shared viral attribute is resolved by CONF_RULE inside the join."""
        result = run(
            script=CONF_RULE + f"DS_r <- {join_op}(DS_1, DS_2);",
            data_structures={"datasets": [DS_JOIN_1, DS_JOIN_2]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "VAt_1": ["C", "N", "F"]}
                ),
                "DS_2": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_2": [5.0, 15.0, 25.0], "VAt_1": ["N", "F", "F"]}
                ),
            },
        )
        ds_r = result["DS_r"]
        # Single combined column, keeping the viral role (not #-qualified per operand).
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        # C+N→C (unary "C"); N+F→N (unary "N"); F+F→F (else)
        sorted_data = ds_r.data.sort_values("Id_1").reset_index(drop=True)
        assert list(sorted_data["VAt_1"]) == ["C", "N", "F"]

    def test_binary_clause_propagation_join(self) -> None:
        """Binary propagation clauses take precedence over unary ones in a join."""
        result = run(
            script=CONF_BINARY_RULE + "DS_r <- left_join(DS_1, DS_2);",
            data_structures={"datasets": [DS_JOIN_1, DS_JOIN_2]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "VAt_1": ["C", "M", "X"]}
                ),
                "DS_2": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_2": [5.0, 15.0, 25.0], "VAt_1": ["M", "F", "Y"]}
                ),
            },
        )
        # C+M→N (binary); M+F→M (unary "M"); X+Y→" " (else)
        sorted_data = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        assert list(sorted_data["VAt_1"]) == ["N", "M", " "]

    def test_enumerated_propagation_cross_join(self) -> None:
        """A viral attribute shared by both cross_join operands is combined via the
        propagation rule (cross_join pairs every row, so use one row per operand)."""
        result = run(
            script=CONF_RULE + "DS_r <- cross_join(DS_1, DS_2);",
            data_structures={"datasets": [DS_JOIN_1, DS_CROSS_2]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["C"]}),
                "DS_2": pd.DataFrame({"Id_2": [2], "Me_2": [5.0], "VAt_1": ["N"]}),
            },
        )
        ds_r = result["DS_r"]
        # Single combined viral column (not #-qualified per operand).
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert "DS_1#VAt_1" not in ds_r.components
        # C+N→C (unary "C")
        assert list(ds_r.data["VAt_1"]) == ["C"]

    @pytest.mark.parametrize("join_op", ["inner_join", "left_join", "full_join"])
    def test_no_rule_gives_null_join(self, join_op: str) -> None:
        """Both operands viral but no rule defined → combined value is null."""
        result = run(
            script=f"DS_r <- {join_op}(DS_1, DS_2);",
            data_structures={"datasets": [DS_JOIN_1, DS_JOIN_2]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]}),
                "DS_2": pd.DataFrame({"Id_1": [1], "Me_2": [5.0], "VAt_1": ["B"]}),
            },
        )
        ds_r = result["DS_r"]
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert pd.isna(ds_r.data["VAt_1"].iloc[0])

    def test_no_rule_gives_null_cross_join(self) -> None:
        """cross_join, both operands viral, no rule defined → combined value null."""
        result = run(
            script="DS_r <- cross_join(DS_1, DS_2);",
            data_structures={"datasets": [DS_JOIN_1, DS_CROSS_2]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]}),
                "DS_2": pd.DataFrame({"Id_2": [1], "Me_2": [5.0], "VAt_1": ["B"]}),
            },
        )
        ds_r = result["DS_r"]
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert pd.isna(ds_r.data["VAt_1"].iloc[0])

    @pytest.mark.parametrize("join_op", ["inner_join", "left_join", "full_join"])
    def test_viral_from_one_operand_kept(self, join_op: str) -> None:
        """A viral attribute present in a single operand is carried over unchanged
        (no propagation rule needed)."""
        result = run(
            script=CONF_RULE + f"DS_r <- {join_op}(DS_1, DS_2);",
            data_structures={"datasets": [DS_JOIN_1, DS_JOIN_2_NO_VA]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["C", "N"]}),
                "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_2": [5.0, 15.0]}),
            },
        )
        ds_r = result["DS_r"]
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert ds_r.data["VAt_1"].notna().all()
        assert set(ds_r.data["VAt_1"]) == {"C", "N"}

    def test_viral_from_one_operand_kept_cross_join(self) -> None:
        """cross_join: a viral attribute present in a single operand is kept
        unchanged (repeated cartesian-wise)."""
        result = run(
            script=CONF_RULE + "DS_r <- cross_join(DS_1, DS_2);",
            data_structures={"datasets": [DS_JOIN_1, DS_CROSS_2_NO_VA]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["C", "N"]}),
                "DS_2": pd.DataFrame({"Id_2": [1, 2], "Me_2": [5.0, 15.0]}),
            },
        )
        ds_r = result["DS_r"]
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert ds_r.data["VAt_1"].notna().all()
        assert set(ds_r.data["VAt_1"]) == {"C", "N"}
