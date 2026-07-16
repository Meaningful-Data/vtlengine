"""Viral propagation rule execution across operators (issues #877, #906).

The propagation rule is executed (and required) ONLY where input data points are
combined: an operation over two or more datasets, an aggregation/analytic group-by,
or a hierarchy roll-up. Row-preserving operators (unary, dataset-scalar, unpivot,
period_indicator, check_datapoint, check_hierarchy) COPY viral attributes unchanged.
Also covers the strict semantic checks 1-3-3-5 (aggregate sum/avg needs a numeric
viral attribute) and 1-3-3-6 (a combined viral attribute requires a rule).
"""

import pandas as pd
import pytest

from vtlengine import run, semantic_analysis
from vtlengine.DataTypes import Integer, String
from vtlengine.Exceptions import SemanticError
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

AGGR_MAX_RULE = """
    define viral propagation S (variable VAt_1) is
        aggregate max
    end viral propagation;
"""

_ENUM_RULE = (
    'define viral propagation VP (variable VAt_1) is when "A" then "Z"; else "D" '
    "end viral propagation;"
)

VP_IDENTITY = (
    'define viral propagation VP_1 (variable VAt_1) is when "A" then "A"; when "B" then "B" '
    "end viral propagation;\n"
)

# -- Shared datasets --


def _single_va_ds(vtype: str = "String") -> dict:
    """DS_1 with one viral attribute of the given type."""
    return {
        "name": "DS_1",
        "DataStructure": [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            {"name": "VAt_1", "type": vtype, "role": "Viral Attribute", "nullable": True},
        ],
    }


DS_1VA = _single_va_ds()

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


# -- Unpivot clause --


class TestViralAttributeUnpivot:
    """Viral attributes must replicate across the rows produced by unpivot (issue #877)."""

    @staticmethod
    def _ds(vtype: str = "String") -> dict:
        return {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "Me_2", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": vtype, "role": "Viral Attribute", "nullable": True},
            ],
        }

    @staticmethod
    def _dp() -> pd.DataFrame:
        return pd.DataFrame(
            {"Id_1": [1, 2], "Me_1": [10.0, 20.0], "Me_2": [100.0, 200.0], "VAt_1": ["A", "B"]}
        )

    def test_unpivot_replicates_viral_attrs(self) -> None:
        result = run(
            script=f"{VP_IDENTITY}DS_r <- DS_1[unpivot Id_2, Val];",
            data_structures={"datasets": [self._ds()]},
            datapoints={"DS_1": self._dp()},
        )
        ds_r = result["DS_r"]
        assert "VAt_1" in ds_r.components, "VAt_1 missing from result components"
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert "VAt_1" in ds_r.data.columns, "VAt_1 missing from result data"
        # Two source measures -> two melted rows per source row, viral value replicated.
        assert len(ds_r.data) == 4
        expected = {1: "A", 2: "B"}
        for _, row in ds_r.data.iterrows():
            assert row["VAt_1"] == expected[row["Id_1"]]

    def test_unpivot_copies_viral_not_aggregate(self) -> None:
        """Unpivot is row-preserving: an aggregate rule must NOT collapse the viral column;
        each source row's value is copied (replicated) across the unpivoted rows (#906)."""
        df = pd.DataFrame(
            {"Id_1": [1, 2], "Me_1": [10.0, 20.0], "Me_2": [100.0, 200.0], "VAt_1": [1, 2]}
        )
        result = run(
            script=f"{AGGR_MAX_RULE}DS_r <- DS_1[unpivot Id_2, Val];",
            data_structures={"datasets": [self._ds("Number")]},
            datapoints={"DS_1": df},
        )
        # Copied per source row (NOT collapsed to the dataset-wide max 2).
        expected = {1: 1, 2: 2}
        for _, row in result["DS_r"].data.iterrows():
            assert row["VAt_1"] == expected[row["Id_1"]]


# -- Period_indicator time operator --


class TestViralAttributePeriodIndicator:
    """Viral attributes must pass through period_indicator unchanged (issue #877)."""

    @staticmethod
    def _ds(vtype: str = "String") -> dict:
        return {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Time_Period", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": vtype, "role": "Viral Attribute", "nullable": True},
            ],
        }

    def test_period_indicator_preserves_viral_attrs(self) -> None:
        df = pd.DataFrame({"Id_1": ["2020-01", "2020-02"], "Me_1": [1.0, 2.0], "VAt_1": ["A", "B"]})
        result = run(
            script=f"{VP_IDENTITY}DS_r <- period_indicator(DS_1);",
            data_structures={"datasets": [self._ds()]},
            datapoints={"DS_1": df},
        )
        ds_r = result["DS_r"]
        assert "VAt_1" in ds_r.components, "VAt_1 missing from result components"
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert "VAt_1" in ds_r.data.columns, "VAt_1 missing from result data"
        # Row-wise op: viral value carried over unchanged, matched to the source time period.
        expected = {"2020M1": "A", "2020M2": "B"}
        for _, row in ds_r.data.iterrows():
            assert row["VAt_1"] == expected[str(row["Id_1"])]

    def test_period_indicator_copies_viral_not_aggregate(self) -> None:
        """period_indicator is row-preserving: an aggregate rule must NOT collapse the
        viral column; each row's value is copied unchanged (issue #906)."""
        df = pd.DataFrame({"Id_1": ["2020-01", "2020-02"], "Me_1": [1.0, 2.0], "VAt_1": [100, 200]})
        result = run(
            script=f"{AGGR_MAX_RULE}DS_r <- period_indicator(DS_1);",
            data_structures={"datasets": [self._ds("Number")]},
            datapoints={"DS_1": df},
        )
        # Copied per row (NOT collapsed to the dataset-wide max 200).
        assert sorted(result["DS_r"].data["VAt_1"].tolist()) == [100, 200]


# -- check_datapoint validation operator --

_DPR = """
define datapoint ruleset R (variable Me_1) is
    r1: when Me_1 > 0 then Me_1 < 15 errorcode "e1" errorlevel 1
end datapoint ruleset;
"""


class TestViralAttributeCheckDatapoint:
    """Viral attributes must be re-attached to the check_datapoint result, per source
    datapoint (issue #877)."""

    @pytest.mark.parametrize("output", ["", "all", "all_measures"])
    def test_check_datapoint_reattaches_viral(self, output: str) -> None:
        expr = f"check_datapoint(DS_1, R {output})" if output else "check_datapoint(DS_1, R)"
        result = run(
            script=f"{VP_IDENTITY}{_DPR}\nDS_r <- {expr};",
            data_structures={"datasets": [_single_va_ds()]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["A", "B"]})
            },
        )
        ds_r = result["DS_r"]
        assert "VAt_1" in ds_r.components, "VAt_1 missing from result components"
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert "VAt_1" in ds_r.data.columns, "VAt_1 missing from result data"
        expected = {1: "A", 2: "B"}
        for _, row in ds_r.data.iterrows():
            assert row["VAt_1"] == expected[row["Id_1"]]

    def test_check_datapoint_copies_viral_not_aggregate(self) -> None:
        """check_datapoint is row-preserving: an aggregate rule must NOT collapse the viral
        column; each source datapoint's value is copied unchanged (issue #906, superseding
        #897's dataset-wide aggregate)."""
        df = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "VAt_1": [100, 200, 300]})
        result = run(
            script=f"{AGGR_MAX_RULE}{_DPR}\nDS_r <- check_datapoint(DS_1, R all);",
            data_structures={"datasets": [_single_va_ds("Number")]},
            datapoints={"DS_1": df},
        )
        # Copied per datapoint (NOT collapsed to the dataset-wide max 300).
        d = result["DS_r"].data.sort_values("Id_1")
        assert list(d["VAt_1"]) == [100, 200, 300]


# -- Rule execution on row-preserving dataset-level operators (issue #877) --


class TestViralRuleExecutionRowPreserving:
    """Row-preserving dataset-level operators must COPY the viral attribute, NOT execute
    the rule: an aggregate rule does not collapse it and an enumerated rule does not remap
    it, because no data points are combined (issue #906)."""

    @pytest.mark.parametrize("expr", ["round(DS_1)", "abs(DS_1)", "DS_1 * 2"])
    def test_aggregate_rule_copies_per_row(self, expr: str) -> None:
        result = run(
            script=f"{AGGR_MAX_RULE}DS_r <- {expr};",
            data_structures={"datasets": [_single_va_ds("Number")]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [1.0, 2.0, 3.0], "VAt_1": [100, 200, 300]}
                )
            },
        )
        # Copied per row (NOT collapsed to the dataset-wide max 300).
        d = result["DS_r"].data.sort_values("Id_1")
        assert list(d["VAt_1"]) == [100, 200, 300]

    def test_enumerated_rule_copies_per_row(self) -> None:
        result = run(
            script=f"{_ENUM_RULE}\nDS_r <- round(DS_1);",
            data_structures={"datasets": [_single_va_ds()]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [1.0, 2.0], "VAt_1": ["A", "B"]})
            },
        )
        df = result["DS_r"].data.sort_values("Id_1")
        # Copied unchanged (NOT remapped "A"->"Z", "B"->"D").
        assert list(df["VAt_1"]) == ["A", "B"]

    def test_dataset_scalar_binary_copies_and_needs_no_rule(self) -> None:
        """A dataset-scalar binary (DS ⊕ scalar) is row-preserving: it copies the viral
        attribute (an aggregate rule must NOT collapse it) and requires no rule (#906)."""
        dp = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [1.0, 2.0, 3.0], "VAt_1": [100, 200, 300]})
        # (a) rule present but not executed -> values copied per row (not the dataset-wide max)
        result = run(
            script=f"{AGGR_MAX_RULE}DS_r <- DS_1 + 5;",
            data_structures={"datasets": [_single_va_ds("Number")]},
            datapoints={"DS_1": dp},
        )
        assert list(result["DS_r"].data.sort_values("Id_1")["VAt_1"]) == [100, 200, 300]
        # (b) no rule at all -> succeeds (no 1-3-3-6), values copied
        result = run(
            script="DS_r <- DS_1 + 5;",
            data_structures={"datasets": [_single_va_ds("Number")]},
            datapoints={"DS_1": dp},
        )
        assert list(result["DS_r"].data.sort_values("Id_1")["VAt_1"]) == [100, 200, 300]


# -- hierarchy aggregation operator --


def _hierarchy_ds() -> dict:
    """DS_1 with a code-item identifier (Id_2) and a numeric viral attribute."""
    return {
        "name": "DS_1",
        "DataStructure": [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            {"name": "VAt_1", "type": "Number", "role": "Viral Attribute", "nullable": True},
        ],
    }


class TestViralAttributeHierarchy:
    """hierarchy computed nodes must combine child viral values via the rule; passthrough
    rows keep their own value (issue #877)."""

    _HR = (
        "define hierarchical ruleset H (valuedomain rule Id_2) is "
        "A = B + C; T = A + D end hierarchical ruleset;"
    )

    @staticmethod
    def _dp() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Id_1": [1, 1, 1],
                "Id_2": ["B", "C", "D"],
                "Me_1": [1.0, 1.0, 1.0],
                "VAt_1": [100, 200, 50],
            }
        )

    def test_hierarchy_combines_child_viral(self) -> None:
        result = run(
            script=f"{AGGR_MAX_RULE}{self._HR}\nDS_r <- hierarchy(DS_1, H rule Id_2 non_null);",
            data_structures={"datasets": [_hierarchy_ds()]},
            datapoints={"DS_1": self._dp()},
        )
        ds_r = result["DS_r"]
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        vat = {row["Id_2"]: row["VAt_1"] for _, row in ds_r.data.iterrows()}
        # A = max(B 100, C 200) = 200; T = max(A 200, D 50) = 200
        assert vat["A"] == 200
        assert vat["T"] == 200


# -- check_hierarchy validation operator --


class TestViralAttributeCheckHierarchy:
    """check_hierarchy re-attaches the validated code item's viral value and executes
    the propagation rule (issue #877)."""

    _HR = (
        "define hierarchical ruleset H (valuedomain rule Id_2) is "
        "A = B + C end hierarchical ruleset;"
    )

    @staticmethod
    def _dp() -> pd.DataFrame:
        # A = 3 == B(1) + C(2), so the A-row validation passes.
        return pd.DataFrame(
            {
                "Id_1": [1, 1, 1],
                "Id_2": ["A", "B", "C"],
                "Me_1": [3.0, 1.0, 2.0],
                "VAt_1": [100, 200, 50],
            }
        )

    def test_check_hierarchy_propagates_viral(self) -> None:
        result = run(
            script=f"{AGGR_MAX_RULE}{self._HR}\nDS_r <- check_hierarchy(DS_1, H rule Id_2 all);",
            data_structures={"datasets": [_hierarchy_ds()]},
            datapoints={"DS_1": self._dp()},
        )
        ds_r = result["DS_r"]
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert "VAt_1" in ds_r.data.columns
        # validation row for A carries A's viral value (100), rule-applied
        rows = ds_r.data[ds_r.data["Id_2"] == "A"]
        assert list(rows["VAt_1"]) == [100]


# -- Aggregate viral-rule type validation (issue #877) --


class TestViralAggregateRuleTypeValidation:
    """`aggregate sum`/`avg` require a numeric viral attribute; a clear SemanticError
    (1-3-3-5) must be raised at semantic analysis time, not a runtime crash."""

    @pytest.mark.parametrize("fn", ["sum", "avg"])
    @pytest.mark.parametrize("vtype", ["String", "Date", "Boolean"])
    def test_sum_avg_non_numeric_raises(self, fn: str, vtype: str) -> None:
        script = (
            f"define viral propagation VP (variable VAt_1) is aggregate {fn} "
            "end viral propagation;\nDS_r <- DS_1 * 2;"
        )
        with pytest.raises(SemanticError) as exc:
            semantic_analysis(script=script, data_structures={"datasets": [_single_va_ds(vtype)]})
        assert "1-3-3-5" in str(exc.value)

    @pytest.mark.parametrize(
        "fn,vtype",
        [("sum", "Number"), ("avg", "Integer"), ("max", "String"), ("min", "Date")],
    )
    def test_valid_combinations_pass(self, fn: str, vtype: str) -> None:
        script = (
            f"define viral propagation VP (variable VAt_1) is aggregate {fn} "
            "end viral propagation;\nDS_r <- DS_1 * 2;"
        )
        # Must not raise.
        semantic_analysis(script=script, data_structures={"datasets": [_single_va_ds(vtype)]})

    @pytest.mark.parametrize("fn", ["sum", "avg"])
    def test_valuedomain_sum_avg_non_numeric_raises(self, fn: str) -> None:
        vd = {"name": "CL_X", "setlist": ["A", "B"], "type": "String"}
        script = (
            f"define viral propagation VP (valuedomain CL_X) is aggregate {fn} "
            "end viral propagation;\nDS_r <- DS_1 * 2;"
        )
        with pytest.raises(SemanticError) as exc:
            semantic_analysis(
                script=script, data_structures={"datasets": [DS_NO_VA]}, value_domains=vd
            )
        assert "1-3-3-5" in str(exc.value)


# -- Every viral attribute requires a rule (issue #877) --


class TestViralRuleRequiredOnlyWhenCombined:
    """Per the VTL 2.2 attribute propagation rule, a rule is required (and executed) only
    where input data points are combined. A viral attribute merely copied through a
    row-preserving / single-operand operator needs no rule (issue #906)."""

    # -- non-combining operators: viral attribute copied through, no rule required --

    def test_calc_measure_no_rule_ok(self) -> None:
        """A calc clause is row-preserving: it copies the viral attribute; no rule needed."""
        result = run(
            script="DS_r <- DS_1[calc Me_2 := Me_1 * 2];",
            data_structures={"datasets": [DS_1VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]})},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(result["DS_r"].data["VAt_1"]) == ["A"]

    def test_unary_no_rule_ok(self) -> None:
        result = run(
            script="DS_r <- abs(DS_1);",
            data_structures={"datasets": [DS_1VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [-10.0], "VAt_1": ["A"]})},
        )
        assert list(result["DS_r"].data["VAt_1"]) == ["A"]

    def test_keep_no_rule_ok(self) -> None:
        result = run(
            script="DS_r <- DS_1[keep Me_1];",
            data_structures={"datasets": [DS_1VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]})},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(result["DS_r"].data["VAt_1"]) == ["A"]

    def test_calc_creates_viral_no_rule_ok(self) -> None:
        result = run(
            script='DS_r <- DS_1[calc viral attribute VAt_1 := "X"];',
            data_structures={"datasets": [DS_NO_VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0]})},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert list(result["DS_r"].data["VAt_1"]) == ["X"]

    def test_two_viral_partial_rule_row_preserving_ok(self) -> None:
        """Two viral attributes, a rule for only one, through a row-preserving operator →
        no error (neither is combined, so even the un-ruled VAt_2 is fine)."""
        # rule only for VAt_1; VAt_2 has none — a row-preserving op combines neither.
        result = run(
            script=AGGR_MAX_RULE + "DS_r <- abs(DS_1);",
            data_structures={"datasets": [DS_2VA]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"], "VAt_2": [7]})
            },
        )
        assert set(result["DS_r"].get_viral_attributes_names()) == {"VAt_1", "VAt_2"}

    def test_semantic_analysis_no_rule_ok(self) -> None:
        """A row-preserving op with no rule passes semantic analysis (no execution)."""
        result = semantic_analysis(
            script="DS_r <- DS_1 * 2;",
            data_structures={"datasets": [DS_1VA]},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE

    # -- combination points: a rule IS required (SemanticError 1-3-3-6) --

    def test_aggregation_no_rule_raises(self) -> None:
        with pytest.raises(SemanticError) as exc:
            semantic_analysis(
                script="DS_r <- sum(DS_1 group by Id_1);",
                data_structures={"datasets": [DS_1VA]},
            )
        assert "1-3-3-6" in str(exc.value)

    def test_analytic_no_rule_raises(self) -> None:
        with pytest.raises(SemanticError) as exc:
            semantic_analysis(
                script="DS_r <- sum(DS_1 over (partition by Id_1));",
                data_structures={"datasets": [DS_1VA]},
            )
        assert "1-3-3-6" in str(exc.value)

    def test_binary_two_datasets_no_rule_raises(self) -> None:
        with pytest.raises(SemanticError) as exc:
            semantic_analysis(
                script="DS_r <- DS_1 + DS_2;",
                data_structures={"datasets": [DS_1VA, {**DS_1VA, "name": "DS_2"}]},
            )
        assert "1-3-3-6" in str(exc.value)

    def test_rule_present_does_not_raise(self) -> None:
        """Positive control: declaring the rule makes a combining script valid."""
        result = run(
            script=CONF_RULE + "DS_r <- DS_1 + DS_2;",
            data_structures={"datasets": [DS_1VA, {**DS_1VA, "name": "DS_2"}]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["C"]}),
                "DS_2": pd.DataFrame({"Id_1": [1], "Me_1": [5.0], "VAt_1": ["N"]}),
            },
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE


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


class TestViralRuleCombinesInGroupAndPartition:
    """At the aggregation and analytic combination points the propagation rule is executed
    within each group / partition (issue #906): an aggregate rule combines the group's
    values (skipping nulls), analytic broadcasts the combined value to every row of the
    partition, and an enumerated rule combines the group's values through its clauses."""

    @pytest.mark.parametrize(
        "agg_fn, group1",
        # group Id_1=1 = {10, null, 30}; nulls are skipped: min 10, max 30, sum 40, avg 20.
        [("min", 10.0), ("max", 30.0), ("sum", 40.0), ("avg", 20.0)],
    )
    def test_aggregate_rule_combines_per_group_skipping_nulls(
        self, agg_fn: str, group1: float
    ) -> None:
        rule = (
            f"define viral propagation S (variable VAt_1) is\n"
            f"    aggregate {agg_fn}\n"
            f"end viral propagation;\n"
        )
        result = run(
            script=rule + "DS_r <- sum(DS_1 group by Id_1);",
            data_structures={"datasets": [_GP_NUM_DS]},
            datapoints={"DS_1": _gp_num_dp()},
        )
        d = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        # One combined value per group; group Id_1=2 keeps its lone value 5.
        assert d["VAt_1"].iloc[0] == group1
        assert d["VAt_1"].iloc[1] == 5.0

    def test_analytic_rule_combines_per_partition_and_broadcasts(self) -> None:
        result = run(
            script=AGGR_MAX_RULE + "DS_r <- sum(DS_1 over (partition by Id_1));",
            data_structures={"datasets": [_GP_NUM_DS]},
            datapoints={"DS_1": _gp_num_dp()},
        )
        d = result["DS_r"].data.sort_values(["Id_1", "Id_2"]).reset_index(drop=True)
        # max over partition Id_1=1 {10, null, 30} = 30, broadcast to all 3 rows; partition 2 = 5.
        assert list(d[d["Id_1"] == 1]["VAt_1"]) == [30.0, 30.0, 30.0]
        assert list(d[d["Id_1"] == 2]["VAt_1"]) == [5.0]

    def test_enumerated_rule_combines_per_group(self) -> None:
        result = run(
            script=_ENUM_PAIR_RULE + "DS_r <- sum(DS_1 group by Id_1);",
            data_structures={"datasets": [_GP_STR_DS]},
            datapoints={"DS_1": _gp_str_dp()},
        )
        d = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        # group {"A","B"} matches the binary clause -> "AB"; group {"C","D"} -> "CD".
        assert list(d["VAt_1"]) == ["AB", "CD"]

    def test_enumerated_rule_combines_per_partition(self) -> None:
        result = run(
            script=_ENUM_PAIR_RULE + "DS_r <- sum(DS_1 over (partition by Id_1));",
            data_structures={"datasets": [_GP_STR_DS]},
            datapoints={"DS_1": _gp_str_dp()},
        )
        d = result["DS_r"].data.sort_values(["Id_1", "Id_2"]).reset_index(drop=True)
        # Combined per partition and broadcast to each row.
        assert list(d[d["Id_1"] == 1]["VAt_1"]) == ["AB", "AB"]
        assert list(d[d["Id_1"] == 2]["VAt_1"]) == ["CD", "CD"]

    @pytest.mark.parametrize(
        "invocation",
        ["sum(DS_1 group by Id_1)", "sum(DS_1 over (partition by Id_1))"],
    )
    def test_enumerated_rule_applies_to_single_element_group(self, invocation: str) -> None:
        """A group / partition with a single data point still has the enumerated rule
        applied to it, exactly as a larger group does — the rule runs in every group
        regardless of size (regression: DuckDB's ``list_reduce`` skipped the lambda for a
        one-element list, leaving the lone value unmapped, issue #906)."""
        rule = (
            "define viral propagation R (variable VAt_1) is\n"
            '    when "A" then "A1";\n'
            '    else "F"\n'
            "end viral propagation;\n"
        )
        # group/partition Id_1=1 -> two rows {"A","A"}; Id_1=2 -> a lone {"A"}.
        dp = pd.DataFrame(
            {
                "Id_1": [1, 1, 2],
                "Id_2": [1, 2, 1],
                "Me_1": [10.0, 20.0, 30.0],
                "VAt_1": ["A", "A", "A"],
            }
        )
        result = run(
            script=rule + f"DS_r <- {invocation};",
            data_structures={"datasets": [_GP_STR_DS]},
            datapoints={"DS_1": dp},
        )
        # Every output row maps "A" -> "A1"; the lone group is NOT copied through as "A".
        assert set(result["DS_r"].data["VAt_1"]) == {"A1"}


# -- DAG statement sorting keeps the rule registered (issue #877) --


class TestViralPropagationDefSurvivesDagSort:
    """Regression: DAGAnalyzer.sort_ast dropped ViralPropagationDef nodes in
    multi-statement scripts, leaving the rule unregistered so combinations silently
    produced NULL (and, with the strict policy, a spurious 1-3-3-6)."""

    def test_rule_survives_multi_statement_script(self) -> None:
        script = f"""
        {AGGR_MAX_RULE}
        DS_A <- DS_1;
        DS_r <- DS_A + DS_A;
        """
        result = run(
            script=script,
            data_structures={"datasets": [DS_1VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]})},
            return_only_persistent=False,
        )
        # The rule is registered and applied: max("A", "A") = "A", not NULL.
        assert list(result["DS_r"].data["VAt_1"]) == ["A"]


# -- Unit tests for the combination-point check helpers (issue #906) --


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
        combined = combined_viral_components(
            [_viral_ds("A", ["VAt_1", "VAt_2"]), _viral_ds("B", ["VAt_1"])]
        )
        assert {c.name for c in combined} == {"VAt_1"}

    def test_combined_viral_components_empty_for_single_operand(self) -> None:
        assert combined_viral_components([_viral_ds("A", ["VAt_1"])]) == []


# -- Row-preserving operators copy the viral attribute, they do NOT execute the rule (#906) --

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


class TestRowPreservingCopiesViral:
    """A rule may be declared, but a row-preserving operator copies the viral attribute
    unchanged: an aggregate rule must NOT collapse it and an enumerated rule must NOT
    remap it, because no data points are combined (issue #906)."""

    def test_unary_aggregate_rule_copies_not_collapses(self) -> None:
        result = run(
            script=AGGR_MAX_RULE + "DS_r <- abs(DS_1);",
            data_structures={"datasets": [NUM_VA_2ID]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {
                        "Id_1": [1, 1, 2],
                        "Id_2": ["A", "B", "A"],
                        "Me_1": [-1.0, -2.0, -3.0],
                        "VAt_1": [10.0, None, 30.0],
                    }
                )
            },
        )
        d = result["DS_r"].data.sort_values(["Id_1", "Id_2"]).reset_index(drop=True)
        # Copied per row, NOT collapsed to the dataset-wide max (30).
        assert d["VAt_1"].iloc[0] == 10.0
        assert pd.isna(d["VAt_1"].iloc[1])
        assert d["VAt_1"].iloc[2] == 30.0

    def test_scalar_enumerated_rule_copies_not_remaps(self) -> None:
        result = run(
            script=ENUM_REMAP_RULE + "DS_r <- DS_1 + 5;",
            data_structures={"datasets": [DS_1VA]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["A", "B"]})
            },
        )
        d = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        # "A" copied (NOT remapped to "Z"); "B" copied (NOT defaulted to "F").
        assert list(d["VAt_1"]) == ["A", "B"]


# -- Propagation rules through join operators (#906) --

CONF_BINARY_RULE = """
    define viral propagation COMP_mix (variable VAt_1) is
        when "C" and "M" then "N";
        when "M" then "M";
        else " "
    end viral propagation;
"""

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
        # C+N->C (unary "C"); N+F->N (unary "N"); F+F->F (else)
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
        # C+M->N (binary); M+F->M (unary "M"); X+Y->" " (else)
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
        # C+N->C (unary "C")
        assert list(ds_r.data["VAt_1"]) == ["C"]

    @pytest.mark.parametrize("join_op", ["inner_join", "left_join", "full_join"])
    def test_no_rule_combine_raises_join(self, join_op: str) -> None:
        """Both operands viral but no rule defined -> SemanticError (issue #877)."""
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
        """cross_join, both operands viral, no rule defined -> SemanticError (issue #877)."""
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
