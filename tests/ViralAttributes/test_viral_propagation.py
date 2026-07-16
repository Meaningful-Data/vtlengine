"""Tests for define viral propagation: parsing, end-to-end execution, and semantic validation."""

from typing import Optional

import pandas as pd
import pytest

from vtlengine import run, semantic_analysis
from vtlengine.API import create_ast
from vtlengine.DataTypes import Integer, String
from vtlengine.Exceptions import SemanticError, VTLSyntaxError
from vtlengine.Model import Component, Dataset, Role
from vtlengine.ViralPropagation import (
    ViralPropagationRegistry,
    ViralPropagationRule,
    combined_viral_components,
    require_rules,
    set_current_registry,
)

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

    def test_no_rule_combine_raises(self) -> None:
        """Both operands viral but no rule defined → SemanticError (issue #877)."""
        with pytest.raises(SemanticError) as exc:
            run(
                script="DS_r <- DS_1 + DS_2;",
                data_structures=_ds_pair(DS_1VA),
                datapoints={
                    "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]}),
                    "DS_2": pd.DataFrame({"Id_1": [1], "Me_1": [5.0], "VAt_1": ["B"]}),
                },
            )
        assert "1-3-3-6" in str(exc.value)

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


# -- A viral attribute requires a rule ONLY when it is combined (issue #906) --


class TestViralRuleRequiredOnlyWhenCombined:
    """Per the VTL 2.2 attribute propagation rule, a rule is required (and executed) only
    where input data points are combined. A viral attribute merely copied through a
    row-preserving / single-operand operator needs no rule."""

    # -- non-combining operators: viral attribute copied through, no rule required --

    def test_identity_assignment_no_rule_ok(self) -> None:
        """``DS_r <- DS_1`` copies the viral attribute; no rule needed."""
        result = run(
            script="DS_r <- DS_1;",
            data_structures={"datasets": [DS_1VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]})},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(result["DS_r"].data["VAt_1"]) == ["A"]

    def test_unary_no_rule_ok(self) -> None:
        """A unary (row-preserving) operator copies the viral attribute; no rule needed."""
        result = run(
            script="DS_r <- abs(DS_1);",
            data_structures={"datasets": [DS_1VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [-10.0], "VAt_1": ["A"]})},
        )
        assert list(result["DS_r"].data["VAt_1"]) == ["A"]

    def test_keep_no_rule_ok(self) -> None:
        """A keep clause carries the viral attribute through unchanged; no rule needed."""
        result = run(
            script="DS_r <- DS_1[keep Me_1];",
            data_structures={"datasets": [DS_1VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]})},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(result["DS_r"].data["VAt_1"]) == ["A"]

    def test_calc_creates_viral_no_rule_ok(self) -> None:
        """A calc that creates a viral attribute (never combined) needs no rule."""
        result = run(
            script='DS_r <- DS_1[calc viral attribute VAt_1 := "X"];',
            data_structures={"datasets": [DS_NO_VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0]})},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(result["DS_r"].data["VAt_1"]) == ["X"]

    def test_two_viral_partial_rule_passthrough_ok(self) -> None:
        """Two viral attributes, a rule for only one, pure passthrough → no error
        (neither is combined)."""
        result = run(
            script=AGGR_MAX_RULE + "DS_r <- DS_1;",  # rule only for VAt_1; VAt_2 has none
            data_structures={"datasets": [DS_2VA]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"], "VAt_2": [7]})
            },
        )
        assert set(result["DS_r"].get_viral_attributes_names()) == {"VAt_1", "VAt_2"}

    def test_semantic_analysis_no_rule_ok(self) -> None:
        """A pure passthrough with no rule passes semantic analysis (no execution)."""
        result = semantic_analysis(
            script="DS_r <- DS_1;",
            data_structures={"datasets": [DS_1VA]},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE

    # -- combination points: a rule IS required (SemanticError 1-3-3-6) --

    def test_aggregation_no_rule_raises(self) -> None:
        """Aggregation combines each group's data points → a rule is required."""
        with pytest.raises(SemanticError) as exc:
            semantic_analysis(
                script="DS_r <- sum(DS_1 group by Id_1);",
                data_structures={"datasets": [DS_1VA]},
            )
        assert "1-3-3-6" in str(exc.value)

    def test_analytic_no_rule_raises(self) -> None:
        """Analytic combines each partition's data points → a rule is required."""
        with pytest.raises(SemanticError) as exc:
            semantic_analysis(
                script="DS_r <- sum(DS_1 over (partition by Id_1));",
                data_structures={"datasets": [DS_1VA]},
            )
        assert "1-3-3-6" in str(exc.value)

    def test_rule_present_does_not_raise(self) -> None:
        """Positive control: declaring the rule makes a combining script valid."""
        result = run(
            script=CONF_RULE + "DS_r <- DS_1 + DS_2;",
            data_structures=_ds_pair(DS_1VA),
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["C"]}),
                "DS_2": pd.DataFrame({"Id_1": [1], "Me_1": [5.0], "VAt_1": ["N"]}),
            },
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE


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
    def test_no_rule_combine_raises_join(self, join_op: str) -> None:
        """Both operands viral but no rule defined → SemanticError (issue #877)."""
        with pytest.raises(SemanticError) as exc:
            run(
                script=f"DS_r <- {join_op}(DS_1, DS_2);",
                data_structures={"datasets": [DS_JOIN_1, DS_JOIN_2]},
                datapoints={
                    "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]}),
                    "DS_2": pd.DataFrame({"Id_1": [1], "Me_2": [5.0], "VAt_1": ["B"]}),
                },
            )
        assert "1-3-3-6" in str(exc.value)

    def test_no_rule_combine_raises_cross_join(self) -> None:
        """cross_join, both operands viral, no rule defined → SemanticError (issue #877)."""
        with pytest.raises(SemanticError) as exc:
            run(
                script="DS_r <- cross_join(DS_1, DS_2);",
                data_structures={"datasets": [DS_JOIN_1, DS_CROSS_2]},
                datapoints={
                    "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]}),
                    "DS_2": pd.DataFrame({"Id_2": [1], "Me_2": [5.0], "VAt_1": ["B"]}),
                },
            )
        assert "1-3-3-6" in str(exc.value)

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


# -- Null operands in the join viral combination --
#
# Regression: the join combined two shared viral columns by dropping nulls
# *before* resolving, so a ``(null, X)`` pair collapsed to ``[X]`` and leaked X
# unchanged instead of going through the propagation rule (which a binary
# operator like ``DS_1 + DS_2`` applied correctly via resolve_pair).

DS_PLUS_2 = {"name": "DS_2", "DataStructure": [_ID_1, _ME_1, _VA]}  # Me_1 in both, for ``+``


class TestViralPropagationJoinNulls:
    """A shared viral attribute where one operand's value is null must still go
    through the propagation rule in a join, exactly like in a binary operator."""

    @pytest.mark.parametrize("use_duckdb", [False, True])
    @pytest.mark.parametrize("join_op", ["inner_join", "left_join", "full_join"])
    def test_null_pair_applies_rule(self, join_op: str, use_duckdb: bool) -> None:
        """(null, X) and (X, null) resolve through the rule, not leaking X."""
        result = run(
            script=CONF_RULE + f"DS_r <- {join_op}(DS_1, DS_2);",
            data_structures={"datasets": [DS_JOIN_1, DS_JOIN_2]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {
                        "Id_1": [1, 2, 3, 4],
                        "Me_1": [1.0, 2.0, 3.0, 4.0],
                        "VAt_1": ["C", "Z", None, None],
                    }
                ),
                "DS_2": pd.DataFrame(
                    {
                        "Id_1": [1, 2, 3, 4],
                        "Me_2": [1.0, 2.0, 3.0, 4.0],
                        "VAt_1": ["N", None, "Z", None],
                    }
                ),
            },
            use_duckdb=use_duckdb,
        )
        d = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        # (C,N)->C (unary); (Z,null)->F (else); (null,Z)->F (else); (null,null)->F (else)
        assert list(d["VAt_1"]) == ["C", "F", "F", "F"]
        # The lone non-null value must NOT survive unchanged.
        assert "Z" not in list(d["VAt_1"])

    @pytest.mark.parametrize("use_duckdb", [False, True])
    def test_join_viral_matches_binary_plus(self, use_duckdb: bool) -> None:
        """The join's viral combination is identical to the binary ``+`` one for
        the same viral data (including null operands). The ``+`` uses a matching
        measure; the join uses distinct measures so the non-key columns do not
        collide on the final un-prefix step."""
        vat_1 = ["C", "Z", None, None, "N"]
        vat_2 = ["N", None, "Z", None, "N"]
        ids = [1, 2, 3, 4, 5]
        nums = [1.0, 2.0, 3.0, 4.0, 5.0]
        plus = run(
            script=CONF_RULE + "DS_r <- DS_1 + DS_2;",
            data_structures={"datasets": [DS_JOIN_1, DS_PLUS_2]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": ids, "Me_1": nums, "VAt_1": vat_1}),
                "DS_2": pd.DataFrame({"Id_1": ids, "Me_1": nums, "VAt_1": vat_2}),
            },
            use_duckdb=use_duckdb,
        )
        join = run(
            script=CONF_RULE + "DS_r <- inner_join(DS_1, DS_2);",
            data_structures={"datasets": [DS_JOIN_1, DS_JOIN_2]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": ids, "Me_1": nums, "VAt_1": vat_1}),
                "DS_2": pd.DataFrame({"Id_1": ids, "Me_2": nums, "VAt_1": vat_2}),
            },
            use_duckdb=use_duckdb,
        )
        p = plus["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        j = join["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        assert list(p["VAt_1"]) == list(j["VAt_1"])

    @pytest.mark.parametrize("use_duckdb", [False, True])
    def test_aggregate_rule_ignores_nulls_in_join(self, use_duckdb: bool) -> None:
        """An aggregate (max) viral rule ignores nulls in a join: (null, X)->X,
        (null, null)->null."""
        result = run(
            script=AGGR_MAX_RULE + "DS_r <- inner_join(DS_1, DS_2);",
            data_structures={"datasets": [DS_JOIN_1, DS_JOIN_2]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [1.0, 2.0, 3.0], "VAt_1": ["B", None, None]}
                ),
                "DS_2": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_2": [1.0, 2.0, 3.0], "VAt_1": [None, "A", None]}
                ),
            },
            use_duckdb=use_duckdb,
        )
        d = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        assert d["VAt_1"].iloc[0] == "B"  # (B, null) -> B
        assert d["VAt_1"].iloc[1] == "A"  # (null, A) -> A
        assert pd.isna(d["VAt_1"].iloc[2])  # (null, null) -> null


# -- keep clause must preserve viral attributes (they always propagate) --

_AT = {"name": "At_1", "type": "String", "role": "Attribute", "nullable": True}
_VA_2 = {"name": "VAt_2", "type": "Integer", "role": "Viral Attribute", "nullable": True}
DS_KEEP = {"name": "DS_1", "DataStructure": [_ID_1, _ME_1, _ME_2, _AT, _VA]}
DS_KEEP_2VA = {"name": "DS_1", "DataStructure": [_ID_1, _ME_1, _VA, _VA_2]}
DS_NA_VA_1 = {"name": "DS_1", "DataStructure": [_ID_1, _ME_1, _AT, _VA]}
DS_NA_VA_2 = {"name": "DS_2", "DataStructure": [_ID_1, _ME_2, _AT, _VA]}


class TestKeepPreservesViralAttributes:
    """A keep clause restricts identifiers/measures/non-viral attributes, but viral
    attributes always propagate and survive implicitly (without being listed)."""

    @pytest.mark.parametrize("use_duckdb", [False, True])
    def test_standalone_keep_preserves_viral_drops_rest(self, use_duckdb: bool) -> None:
        """Keep Me_1 keeps the viral attr but drops the other measure and the
        non-viral attribute."""
        result = run(
            script=CONF_RULE + "DS_r <- DS_1[keep Me_1];",
            data_structures={"datasets": [DS_KEEP]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {
                        "Id_1": [1, 2],
                        "Me_1": [10.0, 20.0],
                        "Me_2": [1.0, 2.0],
                        "At_1": ["a", "b"],
                        "VAt_1": ["C", "N"],
                    }
                )
            },
            use_duckdb=use_duckdb,
        )
        ds = result["DS_r"]
        assert set(ds.components) == {"Id_1", "Me_1", "VAt_1"}
        assert ds.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        # Single operand: viral value passes through unchanged (no combination).
        d = ds.data.sort_values("Id_1").reset_index(drop=True)
        assert "VAt_1" in d.columns
        assert list(d["VAt_1"]) == ["C", "N"]

    @pytest.mark.parametrize("use_duckdb", [False, True])
    def test_keep_listing_viral_no_duplicate(self, use_duckdb: bool) -> None:
        """Listing the viral attribute explicitly in keep does not duplicate it."""
        result = run(
            script=CONF_RULE + "DS_r <- DS_1[keep Me_1, VAt_1];",
            data_structures={"datasets": [DS_KEEP]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {
                        "Id_1": [1],
                        "Me_1": [10.0],
                        "Me_2": [1.0],
                        "At_1": ["a"],
                        "VAt_1": ["C"],
                    }
                )
            },
            use_duckdb=use_duckdb,
        )
        ds = result["DS_r"]
        assert set(ds.components) == {"Id_1", "Me_1", "VAt_1"}
        assert list(ds.data.columns).count("VAt_1") == 1

    @pytest.mark.parametrize("use_duckdb", [False, True])
    def test_keep_preserves_multiple_viral(self, use_duckdb: bool) -> None:
        """All viral attributes survive a keep (each has a declared rule)."""
        result = run(
            script=TWO_RULES + "DS_r <- DS_1[keep Me_1];",
            data_structures={"datasets": [DS_KEEP_2VA]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["C"], "VAt_2": [7]})
            },
            use_duckdb=use_duckdb,
        )
        ds = result["DS_r"]
        assert set(ds.components) == {"Id_1", "Me_1", "VAt_1", "VAt_2"}
        assert ds.components["VAt_2"].role == Role.VIRAL_ATTRIBUTE

    @pytest.mark.parametrize("use_duckdb", [False, True])
    @pytest.mark.parametrize("join_op", ["inner_join", "left_join", "full_join"])
    def test_keep_in_join_preserves_combined_viral(self, join_op: str, use_duckdb: bool) -> None:
        """A keep inside a join keeps the merged viral attribute, combined by the
        rule, even though only a measure is listed."""
        result = run(
            script=CONF_RULE + f"DS_r <- {join_op}(DS_1, DS_2 keep Me_1);",
            data_structures={"datasets": [DS_JOIN_1, DS_JOIN_2]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["C", "N"]}),
                "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_2": [5.0, 15.0], "VAt_1": ["N", "F"]}),
            },
            use_duckdb=use_duckdb,
        )
        ds = result["DS_r"]
        assert set(ds.components) == {"Id_1", "Me_1", "VAt_1"}
        assert ds.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        d = ds.data.sort_values("Id_1").reset_index(drop=True)
        # C+N->C (unary "C"); N+F->N (unary "N")
        assert list(d["VAt_1"]) == ["C", "N"]

    @pytest.mark.parametrize("use_duckdb", [False, True])
    def test_keep_nonviral_attr_in_join_keeps_viral_with_null(self, use_duckdb: bool) -> None:
        """The reported scenario: keeping a non-viral attribute in a join still
        propagates the viral attribute, and a (null, X) pair resolves via the rule."""
        result = run(
            script=CONF_RULE + "DS_r <- inner_join(DS_1, DS_2 keep DS_2#At_1);",
            data_structures={"datasets": [DS_NA_VA_1, DS_NA_VA_2]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": ["x"], "VAt_1": [None]}),
                "DS_2": pd.DataFrame({"Id_1": [1], "Me_2": [5.0], "At_1": ["y"], "VAt_1": ["Z"]}),
            },
            use_duckdb=use_duckdb,
        )
        ds = result["DS_r"]
        # The explicitly kept non-viral attribute and the viral attribute survive.
        assert set(ds.components) == {"Id_1", "At_1", "VAt_1"}
        assert ds.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert ds.components["At_1"].role == Role.ATTRIBUTE
        # (null, "Z") -> "F" (else), not the leaked "Z".
        assert ds.data["VAt_1"].iloc[0] == "F"
        assert ds.data["At_1"].iloc[0] == "y"


# -- Unit tests for the combination-point check helpers --


def _viral_ds(name: str, viral_names: list) -> Dataset:
    comps = {"Id_1": Component("Id_1", Integer, Role.IDENTIFIER, False)}
    for v in viral_names:
        comps[v] = Component(v, String, Role.VIRAL_ATTRIBUTE, True)
    return Dataset(name=name, components=comps, data=None)


class TestViralCheckHelpers:
    """``require_rules`` / ``combined_viral_components`` back the combination-point check."""

    def test_require_rules_raises_when_missing(self) -> None:
        set_current_registry(ViralPropagationRegistry())
        comp = Component("VAt_1", String, Role.VIRAL_ATTRIBUTE, True)
        with pytest.raises(SemanticError) as exc:
            require_rules([comp])
        assert "1-3-3-6" in str(exc.value)

    def test_require_rules_passes_when_present(self) -> None:
        registry = ViralPropagationRegistry()
        registry.register(
            ViralPropagationRule(
                name="VAt_1",
                signature_type="variable",
                target="VAt_1",
                enumerated_clauses=[],
                aggregate_function="max",
            )
        )
        set_current_registry(registry)
        require_rules([Component("VAt_1", String, Role.VIRAL_ATTRIBUTE, True)])  # must not raise

    def test_combined_viral_components_only_shared(self) -> None:
        # VAt_1 is viral in both operands (combined); VAt_2 only in one (copied, no rule needed).
        combined = combined_viral_components(
            [_viral_ds("A", ["VAt_1", "VAt_2"]), _viral_ds("B", ["VAt_1"])]
        )
        assert {c.name for c in combined} == {"VAt_1"}

    def test_combined_viral_components_empty_for_single_operand(self) -> None:
        assert combined_viral_components([_viral_ds("A", ["VAt_1"])]) == []
