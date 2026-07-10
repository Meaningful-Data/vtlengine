"""Viral propagation rule execution across operators (issue #877, ported from 1.9.X).

Covers the operator-level propagation and rule-execution behaviour introduced by
#878: unpivot, period_indicator, check_datapoint, hierarchy and check_hierarchy
propagate viral attributes and execute the propagation rule; row-preserving
operators execute aggregate rules dataset-wide and enumerated rules per row; and
the strict semantic checks 1-3-3-5 (aggregate sum/avg needs a numeric viral
attribute) and 1-3-3-6 (every viral attribute requires a rule).
"""

import pandas as pd
import pytest

from vtlengine import run, semantic_analysis
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

    def test_unpivot_executes_aggregate_rule(self) -> None:
        df = pd.DataFrame(
            {"Id_1": [1, 2], "Me_1": [10.0, 20.0], "Me_2": [100.0, 200.0], "VAt_1": [1, 2]}
        )
        result = run(
            script=f"{AGGR_MAX_RULE}DS_r <- DS_1[unpivot Id_2, Val];",
            data_structures={"datasets": [self._ds("Number")]},
            datapoints={"DS_1": df},
        )
        # aggregate max over the source VAt_1 [1, 2] = 2, replicated to all 4 melted rows
        assert list(result["DS_r"].data["VAt_1"]) == [2, 2, 2, 2]


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

    def test_period_indicator_executes_aggregate_rule(self) -> None:
        df = pd.DataFrame({"Id_1": ["2020-01", "2020-02"], "Me_1": [1.0, 2.0], "VAt_1": [100, 200]})
        result = run(
            script=f"{AGGR_MAX_RULE}DS_r <- period_indicator(DS_1);",
            data_structures={"datasets": [self._ds("Number")]},
            datapoints={"DS_1": df},
        )
        # aggregate max over the whole dataset -> 200 on every result row
        assert list(result["DS_r"].data["VAt_1"]) == [200, 200]


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

    def test_check_datapoint_executes_aggregate_rule(self) -> None:
        df = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "VAt_1": [100, 200, 300]})
        result = run(
            script=f"{AGGR_MAX_RULE}{_DPR}\nDS_r <- check_datapoint(DS_1, R all);",
            data_structures={"datasets": [_single_va_ds("Number")]},
            datapoints={"DS_1": df},
        )
        # aggregate max over all datapoints -> 300 on every validation row
        assert list(result["DS_r"].data["VAt_1"]) == [300, 300, 300]


# -- Rule execution on row-preserving dataset-level operators (issue #877) --


class TestViralRuleExecutionRowPreserving:
    """Row-preserving dataset-level operators must EXECUTE the rule: aggregate rules
    collapse over the whole dataset (one value on every row); enumerated rules map
    per row (issue #877)."""

    @pytest.mark.parametrize("expr", ["round(DS_1)", "abs(DS_1)", "DS_1 * 2"])
    def test_aggregate_rule_is_dataset_wide(self, expr: str) -> None:
        result = run(
            script=f"{AGGR_MAX_RULE}DS_r <- {expr};",
            data_structures={"datasets": [_single_va_ds("Number")]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [1.0, 2.0, 3.0], "VAt_1": [100, 200, 300]}
                )
            },
        )
        # aggregate max over the whole dataset -> 300 on every result row
        assert list(result["DS_r"].data["VAt_1"]) == [300, 300, 300]

    def test_enumerated_rule_is_per_row(self) -> None:
        result = run(
            script=f"{_ENUM_RULE}\nDS_r <- round(DS_1);",
            data_structures={"datasets": [_single_va_ds()]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [1.0, 2.0], "VAt_1": ["A", "B"]})
            },
        )
        df = result["DS_r"].data.sort_values("Id_1")
        # "A" matches the unary clause -> "Z"; "B" is unmatched -> else "D"
        assert list(df["VAt_1"]) == ["Z", "D"]


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


class TestEveryViralAttributeRequiresRule:
    """Strict policy: a viral attribute reaching a result without a ``define viral
    propagation`` rule is a SemanticError (1-3-3-6), even when no combination happens
    (single-operand assignment, unary, keep, calc). A rule is not optional."""

    def test_identity_assignment_no_rule_raises(self) -> None:
        """``DS_r <- DS_1`` with a viral attribute and no rule → error."""
        with pytest.raises(SemanticError) as exc:
            run(
                script="DS_r <- DS_1;",
                data_structures={"datasets": [DS_1VA]},
                datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]})},
            )
        assert "1-3-3-6" in str(exc.value)

    def test_unary_no_rule_raises(self) -> None:
        """A unary (row-preserving) operator over a viral attribute with no rule → error."""
        with pytest.raises(SemanticError) as exc:
            run(
                script="DS_r <- abs(DS_1);",
                data_structures={"datasets": [DS_1VA]},
                datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [-10.0], "VAt_1": ["A"]})},
            )
        assert "1-3-3-6" in str(exc.value)

    def test_keep_no_rule_raises(self) -> None:
        """A keep clause carries the viral attribute through; without a rule → error."""
        with pytest.raises(SemanticError) as exc:
            run(
                script="DS_r <- DS_1[keep Me_1];",
                data_structures={"datasets": [DS_1VA]},
                datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]})},
            )
        assert "1-3-3-6" in str(exc.value)

    def test_calc_creates_viral_no_rule_raises(self) -> None:
        """A calc that creates a viral attribute must also declare a rule for it."""
        with pytest.raises(SemanticError) as exc:
            run(
                script='DS_r <- DS_1[calc viral attribute VAt_1 := "X"];',
                data_structures={"datasets": [DS_NO_VA]},
                datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0]})},
            )
        assert "1-3-3-6" in str(exc.value)

    def test_partial_rules_missing_one_raises(self) -> None:
        """Two viral attributes but only one rule → the un-ruled one still errors."""
        with pytest.raises(SemanticError) as exc:
            run(
                script=AGGR_MAX_RULE + "DS_r <- DS_1;",  # rule only for VAt_1
                data_structures={"datasets": [DS_2VA]},
                datapoints={
                    "DS_1": pd.DataFrame(
                        {"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"], "VAt_2": [7]}
                    )
                },
            )
        # VAt_2 has no rule.
        assert "1-3-3-6" in str(exc.value)
        assert "VAt_2" in str(exc.value)

    def test_semantic_analysis_no_rule_raises(self) -> None:
        """The rule requirement is enforced at semantic-analysis time (no execution)."""
        with pytest.raises(SemanticError) as exc:
            semantic_analysis(
                script="DS_r <- DS_1;",
                data_structures={"datasets": [DS_1VA]},
            )
        assert "1-3-3-6" in str(exc.value)

    def test_rule_present_does_not_raise(self) -> None:
        """Positive control: declaring the rule makes the same script valid."""
        result = run(
            script=CONF_RULE + "DS_r <- DS_1;",
            data_structures={"datasets": [DS_1VA]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["C"]})},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE


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
