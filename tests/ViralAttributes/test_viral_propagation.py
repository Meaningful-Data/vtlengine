"""Tests for define viral propagation: parsing, end-to-end execution, and semantic validation."""

from typing import Optional

import pandas as pd
import pytest

from vtlengine import run
from vtlengine.DataTypes import Integer, String
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset, Role

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

    def test_null_condition_matches_null_value(self) -> None:
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
    def test_aggregate_binary_with_nulls(
        self, agg: str, both_present: int, one_null: Optional[int]
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


# -- The rule COMBINES per group / partition at aggregation & analytic (issue #906) --

# Two identifiers; group/partition Id_1=1 -> {10, null, 30}; Id_1=2 -> {5}.
_GP_NUM_DS = {
    "name": "DS_1",
    "DataStructure": [
        {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
        {"name": "Id_2", "type": "Integer", "role": "Identifier", "nullable": False},
        {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
        {"name": "VAt_1", "type": "Number", "role": "Viral Attribute", "nullable": True},
    ],
}
_GP_STR_DS = {
    **_GP_NUM_DS,
    "DataStructure": [
        *_GP_NUM_DS["DataStructure"][:3],
        {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True},
    ],
}

_ENUM_PAIR_RULE = """
    define viral propagation R (variable VAt_1) is
        when "A" and "B" then "AB";
        when "C" and "D" then "CD";
        else "F"
    end viral propagation;
"""


def _gp_num_dp() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id_1": [1, 1, 1, 2],
            "Id_2": [1, 2, 3, 1],
            "Me_1": [10.0, 20.0, 30.0, 40.0],
            "VAt_1": [10.0, None, 30.0, 5.0],
        }
    )


def _gp_str_dp() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id_1": [1, 1, 2, 2],
            "Id_2": [1, 2, 1, 2],
            "Me_1": [10.0, 20.0, 30.0, 40.0],
            "VAt_1": ["A", "B", "C", "D"],
        }
    )


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

    @pytest.mark.parametrize("join_op", ["inner_join", "left_join", "full_join"])
    def test_null_pair_applies_rule(self, join_op: str) -> None:
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
        )
        d = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        # (C,N)->C (unary); (Z,null)->F (else); (null,Z)->F (else); (null,null)->F (else)
        assert list(d["VAt_1"]) == ["C", "F", "F", "F"]
        # The lone non-null value must NOT survive unchanged.
        assert "Z" not in list(d["VAt_1"])

    def test_join_viral_matches_binary_plus(self) -> None:
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
        )
        join = run(
            script=CONF_RULE + "DS_r <- inner_join(DS_1, DS_2);",
            data_structures={"datasets": [DS_JOIN_1, DS_JOIN_2]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": ids, "Me_1": nums, "VAt_1": vat_1}),
                "DS_2": pd.DataFrame({"Id_1": ids, "Me_2": nums, "VAt_1": vat_2}),
            },
        )
        p = plus["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        j = join["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        assert list(p["VAt_1"]) == list(j["VAt_1"])

    def test_aggregate_rule_ignores_nulls_in_join(self) -> None:
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

    def test_standalone_keep_preserves_viral_drops_rest(self) -> None:
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
        )
        ds = result["DS_r"]
        assert set(ds.components) == {"Id_1", "Me_1", "VAt_1"}
        assert ds.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        # Single operand: viral value passes through unchanged (no combination).
        d = ds.data.sort_values("Id_1").reset_index(drop=True)
        assert "VAt_1" in d.columns
        assert list(d["VAt_1"]) == ["C", "N"]

    def test_keep_listing_viral_no_duplicate(self) -> None:
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
        )
        ds = result["DS_r"]
        assert set(ds.components) == {"Id_1", "Me_1", "VAt_1"}
        assert list(ds.data.columns).count("VAt_1") == 1

    def test_keep_preserves_multiple_viral(self) -> None:
        """All viral attributes survive a keep (each has a declared rule)."""
        result = run(
            script=TWO_RULES + "DS_r <- DS_1[keep Me_1];",
            data_structures={"datasets": [DS_KEEP_2VA]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["C"], "VAt_2": [7]})
            },
        )
        ds = result["DS_r"]
        assert set(ds.components) == {"Id_1", "Me_1", "VAt_1", "VAt_2"}
        assert ds.components["VAt_2"].role == Role.VIRAL_ATTRIBUTE

    @pytest.mark.parametrize("join_op", ["inner_join", "left_join", "full_join"])
    def test_keep_in_join_preserves_combined_viral(self, join_op: str) -> None:
        """A keep inside a join keeps the merged viral attribute, combined by the
        rule, even though only a measure is listed."""
        result = run(
            script=CONF_RULE + f"DS_r <- {join_op}(DS_1, DS_2 keep Me_1);",
            data_structures={"datasets": [DS_JOIN_1, DS_JOIN_2]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["C", "N"]}),
                "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_2": [5.0, 15.0], "VAt_1": ["N", "F"]}),
            },
        )
        ds = result["DS_r"]
        assert set(ds.components) == {"Id_1", "Me_1", "VAt_1"}
        assert ds.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        d = ds.data.sort_values("Id_1").reset_index(drop=True)
        # C+N->C (unary "C"); N+F->N (unary "N")
        assert list(d["VAt_1"]) == ["C", "N"]

    def test_keep_nonviral_attr_in_join_keeps_viral_with_null(self) -> None:
        """The reported scenario: keeping a non-viral attribute in a join still
        propagates the viral attribute, and a (null, X) pair resolves via the rule."""
        result = run(
            script=CONF_RULE + "DS_r <- inner_join(DS_1, DS_2 keep DS_2#At_1);",
            data_structures={"datasets": [DS_NA_VA_1, DS_NA_VA_2]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "At_1": ["x"], "VAt_1": [None]}),
                "DS_2": pd.DataFrame({"Id_1": [1], "Me_2": [5.0], "At_1": ["y"], "VAt_1": ["Z"]}),
            },
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


# -- Row-preserving operators copy viral attributes, they do NOT execute the rule (#906) --

NUM_VA_2ID = {
    "name": "DS_1",
    "DataStructure": [
        {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
        {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
        {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
        {"name": "VAt_1", "type": "Number", "role": "Viral Attribute", "nullable": True},
    ],
}

ENUM_REMAP_RULE = """
    define viral propagation R (variable VAt_1) is
        when "A" then "Z";
        else "F"
    end viral propagation;
"""
