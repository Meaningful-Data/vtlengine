"""Tests for viral attribute propagation through all operator categories."""

import pandas as pd
import pytest

from vtlengine import run, semantic_analysis
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Role

# -- Layered dataset builders --

BASE_COMPS = [
    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
]

VA_1 = {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True}
VA_2 = {"name": "VAt_2", "type": "String", "role": "Viral Attribute", "nullable": True}
VA_3 = {"name": "VAt_3", "type": "String", "role": "Viral Attribute", "nullable": True}

VA_COMPONENTS = [VA_1, VA_2, VA_3]
VA_NAMES = ["VAt_1", "VAt_2", "VAt_3"]
VA_VALUES = [["A", "B"], ["X", "Y"], ["P", "Q"]]


def _make_ds(name: str, num_viral: int) -> dict:
    """Build a dataset definition with 0..3 viral attributes."""
    comps = BASE_COMPS + VA_COMPONENTS[:num_viral]
    return {"name": name, "DataStructure": comps}


def _make_dp(num_viral: int) -> pd.DataFrame:
    """Build datapoints matching a dataset with num_viral viral attributes."""
    data: dict = {"Id_1": [1, 2], "Me_1": [10.0, 20.0]}
    for i in range(num_viral):
        data[VA_NAMES[i]] = VA_VALUES[i]
    return pd.DataFrame(data)


def _run_single(expr: str, num_viral: int) -> dict:
    """Run an expression with a single dataset (DS_1)."""
    return run(
        script=f"DS_r <- {expr};",
        data_structures={"datasets": [_make_ds("DS_1", num_viral)]},
        datapoints={"DS_1": _make_dp(num_viral)},
    )


def _run_pair(expr: str, num_viral: int) -> dict:
    """Run an expression with two datasets (DS_1, DS_2)."""
    return run(
        script=f"DS_r <- {expr};",
        data_structures={"datasets": [_make_ds("DS_1", num_viral), _make_ds("DS_2", num_viral)]},
        datapoints={"DS_1": _make_dp(num_viral), "DS_2": _make_dp(num_viral)},
    )


def _assert_viral_attrs(result: dict, num_viral: int) -> None:
    """Assert that the expected viral attributes are present with correct role."""
    ds_r = result["DS_r"]
    for va_name in VA_NAMES[:num_viral]:
        assert va_name in ds_r.components, f"{va_name} missing from result components"
        assert ds_r.components[va_name].role == Role.VIRAL_ATTRIBUTE


def _assert_component_data_parity(result: dict) -> None:
    """Assert the result data columns match the declared components exactly."""
    ds_r = result["DS_r"]
    assert set(ds_r.data.columns) == set(ds_r.components), (
        f"component/data mismatch: components={sorted(ds_r.components)}, "
        f"data={sorted(ds_r.data.columns)}"
    )


# -- Unary operators --

unary_params = [
    "abs(DS_1)",
    "ceil(DS_1)",
    "floor(DS_1)",
    "sqrt(DS_1)",
    "ln(DS_1)",
    "exp(DS_1)",
    "isnull(DS_1)",
]


class TestViralAttributeUnaryOps:
    @pytest.mark.parametrize("expr", unary_params)
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_unary_preserves_viral_attrs(self, expr: str, num_viral: int) -> None:
        result = _run_single(expr, num_viral)
        _assert_viral_attrs(result, num_viral)


# -- Binary operators (two datasets) --

binary_params = [
    "DS_1 + DS_2",
    "DS_1 - DS_2",
    "DS_1 * DS_2",
    "DS_1 > DS_2",
    "DS_1 = DS_2",
]


class TestViralAttributeBinaryOps:
    @pytest.mark.parametrize("expr", binary_params)
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_binary_preserves_viral_attrs(self, expr: str, num_viral: int) -> None:
        result = _run_pair(expr, num_viral)
        _assert_viral_attrs(result, num_viral)


# -- Binary operators (dataset + scalar) --

binary_scalar_params = [
    "DS_1 + 5",
    "DS_1 * 2",
    "DS_1 - 1",
]


class TestViralAttributeBinaryScalarOps:
    @pytest.mark.parametrize("expr", binary_scalar_params)
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_binary_scalar_preserves_viral_attrs(self, expr: str, num_viral: int) -> None:
        result = _run_single(expr, num_viral)
        _assert_viral_attrs(result, num_viral)


# -- Other operators --

other_single_params = [
    "between(DS_1, 5, 25)",
]


class TestViralAttributeOtherOps:
    @pytest.mark.parametrize("expr", other_single_params)
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_other_single_preserves_viral_attrs(self, expr: str, num_viral: int) -> None:
        result = _run_single(expr, num_viral)
        _assert_viral_attrs(result, num_viral)

    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_intersect_preserves_viral_attrs(self, num_viral: int) -> None:
        result = _run_pair("intersect(DS_1, DS_2)", num_viral)
        _assert_viral_attrs(result, num_viral)

    @pytest.mark.parametrize("agg_op", ["sum", "avg", "count", "min", "max"])
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_aggregation_preserves_viral_attrs(self, agg_op: str, num_viral: int) -> None:
        comps = [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Id_2", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
        ] + VA_COMPONENTS[:num_viral]
        data: dict = {
            "Id_1": [1, 1, 2],
            "Id_2": [1, 2, 1],
            "Me_1": [10.0, 20.0, 30.0],
        }
        for i in range(num_viral):
            data[VA_NAMES[i]] = [VA_VALUES[i][0], VA_VALUES[i][0], VA_VALUES[i][1]]
        result = run(
            script=f"DS_r <- {agg_op}(DS_1 group by Id_1);",
            data_structures={"datasets": [{"name": "DS_1", "DataStructure": comps}]},
            datapoints={"DS_1": pd.DataFrame(data)},
        )
        _assert_viral_attrs(result, num_viral)
        for va_name in VA_NAMES[:num_viral]:
            assert va_name in result["DS_r"].data.columns, f"{va_name} missing from result data"

    @pytest.mark.parametrize(
        "expr",
        [
            "DS_1[aggr Me_3 := count() group by Id_1]",
            "DS_1[aggr Me_3 := sum(Me_1) group by Id_1]",
            "DS_1[aggr Me_3 := avg(Me_1) group by Id_1]",
            "DS_1[aggr Me_3 := min(Me_1) group by Id_1]",
            "DS_1[aggr Me_3 := max(Me_1) group by Id_1]",
        ],
    )
    @pytest.mark.parametrize("use_duckdb", [False, True])
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_aggr_clause_preserves_viral_attrs(
        self, expr: str, num_viral: int, use_duckdb: bool
    ) -> None:
        comps = [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Id_2", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
        ] + VA_COMPONENTS[:num_viral]
        data: dict = {
            "Id_1": [1, 1, 2],
            "Id_2": [1, 2, 1],
            "Me_1": [10.0, 20.0, 30.0],
        }
        for i in range(num_viral):
            data[VA_NAMES[i]] = [VA_VALUES[i][0], VA_VALUES[i][0], VA_VALUES[i][1]]
        result = run(
            script=f"DS_r <- {expr};",
            data_structures={"datasets": [{"name": "DS_1", "DataStructure": comps}]},
            datapoints={"DS_1": pd.DataFrame(data)},
            use_duckdb=use_duckdb,
        )
        _assert_viral_attrs(result, num_viral)
        # The viral attribute column must survive the aggr clause (component/data parity).
        for va_name in VA_NAMES[:num_viral]:
            assert va_name in result["DS_r"].data.columns, f"{va_name} missing from result data"


# -- Conditional operators (if-then-else) --


class TestViralAttributeConditionalOps:
    """Viral attributes must survive an if-then-else whose condition is a dataset.

    The condition dataset (e.g. ``DS_1#Id_2 = "A"``) also carries the viral
    attribute, which previously collided on merge with the branch operands and
    corrupted the result (component/data mismatch -> downstream crash). The
    viral attribute itself combines across branches like a binary operator: with
    no propagation rule its combined value is NULL."""

    @pytest.mark.parametrize("use_duckdb", [False, True])
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_if_dataset_condition_preserves_viral_attrs(
        self, num_viral: int, use_duckdb: bool
    ) -> None:
        comps = [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
        ] + VA_COMPONENTS[:num_viral]
        data: dict = {
            "Id_1": [1, 1, 2],
            "Id_2": ["A", "B", "A"],
            "Me_1": [10.0, 20.0, 30.0],
        }
        for i in range(num_viral):
            data[VA_NAMES[i]] = [VA_VALUES[i][0], VA_VALUES[i][0], VA_VALUES[i][1]]
        result = run(
            script='DS_r <- if DS_1#Id_2 = "A" then DS_1 else DS_1;',
            data_structures={"datasets": [{"name": "DS_1", "DataStructure": comps}]},
            datapoints={"DS_1": pd.DataFrame(data)},
            use_duckdb=use_duckdb,
        )
        _assert_viral_attrs(result, num_viral)
        _assert_component_data_parity(result)
        for i in range(num_viral):
            va = VA_NAMES[i]
            assert result["DS_r"].data[va].isna().all()

    @pytest.mark.parametrize("use_duckdb", [False, True])
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_count_over_if_dataset_condition(self, num_viral: int, use_duckdb: bool) -> None:
        """count() over an if-then-else result must not crash on viral attrs."""
        comps = [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
        ] + VA_COMPONENTS[:num_viral]
        data: dict = {
            "Id_1": [1, 1, 2],
            "Id_2": ["A", "B", "A"],
            "Me_1": [10.0, 20.0, 30.0],
        }
        for i in range(num_viral):
            data[VA_NAMES[i]] = [VA_VALUES[i][0], VA_VALUES[i][0], VA_VALUES[i][1]]
        result = run(
            script='DS_r <- count(if DS_1#Id_2 = "A" then DS_1 else DS_1 group by Id_1);',
            data_structures={"datasets": [{"name": "DS_1", "DataStructure": comps}]},
            datapoints={"DS_1": pd.DataFrame(data)},
            use_duckdb=use_duckdb,
        )
        _assert_viral_attrs(result, num_viral)
        for va_name in VA_NAMES[:num_viral]:
            assert va_name in result["DS_r"].data.columns, f"{va_name} missing from result data"

    @pytest.mark.parametrize("use_duckdb", [False, True])
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_case_dataset_condition_preserves_viral_attrs(
        self, num_viral: int, use_duckdb: bool
    ) -> None:
        """A dataset-level ``case`` keeps viral attrs (1:1) with no phantom columns."""
        comps = [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
        ] + VA_COMPONENTS[:num_viral]
        data: dict = {
            "Id_1": [1, 1, 2],
            "Id_2": ["A", "B", "A"],
            "Me_1": [10.0, 20.0, 30.0],
        }
        for i in range(num_viral):
            data[VA_NAMES[i]] = [VA_VALUES[i][0], VA_VALUES[i][0], VA_VALUES[i][1]]
        result = run(
            script='DS_r <- case when DS_1#Id_2 = "A" then DS_1 else DS_1;',
            data_structures={"datasets": [{"name": "DS_1", "DataStructure": comps}]},
            datapoints={"DS_1": pd.DataFrame(data)},
            use_duckdb=use_duckdb,
        )
        _assert_viral_attrs(result, num_viral)
        _assert_component_data_parity(result)

    @pytest.mark.parametrize("use_duckdb", [False, True])
    @pytest.mark.parametrize(
        "expr",
        ["nvl(DS_1, DS_2)", "nvl(DS_1, 0.0)"],
    )
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_nvl_preserves_viral_attrs(self, expr: str, num_viral: int, use_duckdb: bool) -> None:
        comps = BASE_COMPS + VA_COMPONENTS[:num_viral]
        structures = [{"name": "DS_1", "DataStructure": comps}]
        datapoints = {"DS_1": _make_dp(num_viral)}
        if "DS_2" in expr:
            structures.append({"name": "DS_2", "DataStructure": comps})
            datapoints["DS_2"] = _make_dp(num_viral)
        result = run(
            script=f"DS_r <- {expr};",
            data_structures={"datasets": structures},
            datapoints=datapoints,
            use_duckdb=use_duckdb,
        )
        _assert_viral_attrs(result, num_viral)
        _assert_component_data_parity(result)


# -- String parameterized operators (substr, replace, instr) --


class TestViralAttributeStringParameterizedOps:
    """Viral attributes must keep BOTH their component role and their data values
    through parameterized string operators (substr, replace, instr)."""

    @staticmethod
    def _string_ds(num_viral: int) -> dict:
        comps = [
            {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
            {"name": "Me_1", "type": "String", "role": "Measure", "nullable": True},
        ] + VA_COMPONENTS[:num_viral]
        return {"name": "DS_1", "DataStructure": comps}

    @staticmethod
    def _string_dp(num_viral: int) -> pd.DataFrame:
        data: dict = {"Id_1": [1, 2], "Me_1": ["hello", "world"]}
        for i in range(num_viral):
            data[VA_NAMES[i]] = VA_VALUES[i]
        return pd.DataFrame(data)

    @pytest.mark.parametrize(
        "expr",
        [
            "substr(DS_1, 2)",
            "substr(DS_1, 2, 3)",
            'replace(DS_1, "l", "L")',
            'instr(DS_1, "o")',
        ],
    )
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_string_param_preserves_viral_attrs(self, expr: str, num_viral: int) -> None:
        result = run(
            script=f"DS_r <- {expr};",
            data_structures={"datasets": [self._string_ds(num_viral)]},
            datapoints={"DS_1": self._string_dp(num_viral)},
        )
        _assert_viral_attrs(result, num_viral)
        # The viral attribute data values must be carried over unchanged (issue #782).
        for i in range(num_viral):
            va_name = VA_NAMES[i]
            assert va_name in result["DS_r"].data.columns, f"{va_name} missing from result data"
            assert list(result["DS_r"].data[va_name]) == VA_VALUES[i]


# -- Numeric parameterized operators (round, trunc) --


class TestViralAttributeNumericParameterizedOps:
    """Viral attributes must keep BOTH their component role and their data values
    through parameterized numeric operators (round, trunc).

    Regression (issue #833): ``Parameterized.dataset_evaluation`` rebuilt the
    result data with only identifiers and measures, so the viral attribute was
    kept in ``result.components`` but dropped from ``result.data`` (component/data
    mismatch)."""

    @pytest.mark.parametrize(
        "expr",
        [
            "round(DS_1, 2)",
            "round(DS_1)",
            "trunc(DS_1, 1)",
            "trunc(DS_1)",
        ],
    )
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_numeric_param_preserves_viral_attrs(self, expr: str, num_viral: int) -> None:
        result = _run_single(expr, num_viral)
        _assert_viral_attrs(result, num_viral)
        _assert_component_data_parity(result)
        # The viral attribute data values must be carried over unchanged.
        for i in range(num_viral):
            va_name = VA_NAMES[i]
            assert va_name in result["DS_r"].data.columns, f"{va_name} missing from result data"
            assert list(result["DS_r"].data[va_name]) == VA_VALUES[i]


# -- Set comparison operators (in / not_in) --


class TestViralAttributeCompOps:
    """Viral attributes must keep BOTH their component role and their data values
    through the operators ``in`` and ``not_in``."""

    @pytest.mark.parametrize("expr", ["DS_1 in {10, 30}", "DS_1 not_in {10, 30}"])
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_set_membership_preserves_viral_attrs(self, expr: str, num_viral: int) -> None:
        result = _run_single(expr, num_viral)
        _assert_viral_attrs(result, num_viral)
        # The viral attribute data values must be carried over unchanged.
        for i in range(num_viral):
            va_name = VA_NAMES[i]
            assert va_name in result["DS_r"].data.columns, f"{va_name} missing from result data"
            assert list(result["DS_r"].data[va_name]) == VA_VALUES[i]


# -- Special cases --


class TestViralAttributeSpecialCases:
    def test_non_viral_attribute_still_dropped(self) -> None:
        ds = {
            "name": "DS_1",
            "DataStructure": [
                *BASE_COMPS,
                {"name": "VAt_1", "type": "String", "role": "Attribute", "nullable": True},
            ],
        }
        result = run(
            script="DS_r <- DS_1 + DS_2;",
            data_structures={"datasets": [ds, {"name": "DS_2", "DataStructure": BASE_COMPS}]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["A", "B"]}),
                "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 15.0]}),
            },
        )
        assert "VAt_1" not in result["DS_r"].components

    def test_calc_viral_attribute(self) -> None:
        result = run(
            script='DS_r <- DS_1 [calc viral attribute VAt_1 := "X"];',
            data_structures={"datasets": [{"name": "DS_1", "DataStructure": BASE_COMPS}]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0]})},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE

    def test_input_viral_attribute_legacy_format(self) -> None:
        ds = {
            "name": "DS_1",
            "DataStructure": [
                *BASE_COMPS,
                {"name": "VAt_1", "type": "String", "role": "ViralAttribute", "nullable": True},
            ],
        }
        result = run(
            script="DS_r <- DS_1;",
            data_structures={"datasets": [ds]},
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [10.0], "VAt_1": ["A"]})},
        )
        assert result["DS_r"].components["VAt_1"].role == Role.VIRAL_ATTRIBUTE

    def test_binary_one_operand_viral(self) -> None:
        """Only DS_1 has viral attr, DS_2 doesn't — viral attr propagated from DS_1."""
        result = run(
            script="DS_r <- DS_1 + DS_2;",
            data_structures={"datasets": [_make_ds("DS_1", 2), _make_ds("DS_2", 0)]},
            datapoints={
                "DS_1": _make_dp(2),
                "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [5.0, 15.0]}),
            },
        )
        _assert_viral_attrs(result, 2)
        assert list(result["DS_r"].data["VAt_1"]) == ["A", "B"]
        assert list(result["DS_r"].data["VAt_2"]) == ["X", "Y"]


# -- Backends exercised by the operator-specific suites below (issue #877) --

BACKENDS = [False, True]


# -- Unpivot clause --


class TestViralAttributeUnpivot:
    """Viral attributes must replicate across the rows produced by unpivot (issue #877)."""

    @staticmethod
    def _ds() -> dict:
        return {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "Me_2", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True},
            ],
        }

    @staticmethod
    def _dp() -> pd.DataFrame:
        return pd.DataFrame(
            {"Id_1": [1, 2], "Me_1": [10.0, 20.0], "Me_2": [100.0, 200.0], "VAt_1": ["A", "B"]}
        )

    @pytest.mark.parametrize("use_duckdb", BACKENDS)
    def test_unpivot_replicates_viral_attrs(self, use_duckdb: bool) -> None:
        result = run(
            script="DS_r <- DS_1[unpivot Id_2, Val];",
            data_structures={"datasets": [self._ds()]},
            datapoints={"DS_1": self._dp()},
            use_duckdb=use_duckdb,
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

    @pytest.mark.parametrize("use_duckdb", BACKENDS)
    def test_unpivot_executes_aggregate_rule(self, use_duckdb: bool) -> None:
        vp = "define viral propagation VP (variable VAt_1) is aggregate max end viral propagation;"
        ds = {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "Me_2", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "Number", "role": "Viral Attribute", "nullable": True},
            ],
        }
        df = pd.DataFrame(
            {"Id_1": [1, 2], "Me_1": [10.0, 20.0], "Me_2": [100.0, 200.0], "VAt_1": [1, 2]}
        )
        result = run(
            script=f"{vp}\nDS_r <- DS_1[unpivot Id_2, Val];",
            data_structures={"datasets": [ds]},
            datapoints={"DS_1": df},
            use_duckdb=use_duckdb,
        )
        # aggregate max over the source VAt_1 [1, 2] = 2, replicated to all 4 melted rows
        assert list(result["DS_r"].data["VAt_1"]) == [2, 2, 2, 2]


# -- Period_indicator time operator --


class TestViralAttributePeriodIndicator:
    """Viral attributes must pass through period_indicator unchanged (issue #877)."""

    @staticmethod
    def _ds() -> dict:
        return {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Time_Period", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True},
            ],
        }

    @staticmethod
    def _dp() -> pd.DataFrame:
        return pd.DataFrame(
            {"Id_1": ["2020-01", "2020-02"], "Me_1": [1.0, 2.0], "VAt_1": ["A", "B"]}
        )

    @pytest.mark.parametrize("use_duckdb", BACKENDS)
    def test_period_indicator_preserves_viral_attrs(self, use_duckdb: bool) -> None:
        result = run(
            script="DS_r <- period_indicator(DS_1);",
            data_structures={"datasets": [self._ds()]},
            datapoints={"DS_1": self._dp()},
            use_duckdb=use_duckdb,
        )
        ds_r = result["DS_r"]
        assert "VAt_1" in ds_r.components, "VAt_1 missing from result components"
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert "VAt_1" in ds_r.data.columns, "VAt_1 missing from result data"
        # Row-wise op: viral value carried over unchanged, matched to the source time period.
        expected = {"2020M1": "A", "2020M2": "B"}
        for _, row in ds_r.data.iterrows():
            assert row["VAt_1"] == expected[str(row["Id_1"])]

    @pytest.mark.parametrize("use_duckdb", BACKENDS)
    def test_period_indicator_executes_aggregate_rule(self, use_duckdb: bool) -> None:
        vp = "define viral propagation VP (variable VAt_1) is aggregate max end viral propagation;"
        ds = {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Time_Period", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "Number", "role": "Viral Attribute", "nullable": True},
            ],
        }
        df = pd.DataFrame({"Id_1": ["2020-01", "2020-02"], "Me_1": [1.0, 2.0], "VAt_1": [100, 200]})
        result = run(
            script=f"{vp}\nDS_r <- period_indicator(DS_1);",
            data_structures={"datasets": [ds]},
            datapoints={"DS_1": df},
            use_duckdb=use_duckdb,
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

    @staticmethod
    def _ds() -> dict:
        return {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "String", "role": "Viral Attribute", "nullable": True},
            ],
        }

    @staticmethod
    def _dp() -> pd.DataFrame:
        return pd.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0], "VAt_1": ["A", "B"]})

    @pytest.mark.parametrize("output", ["", "all", "all_measures"])
    @pytest.mark.parametrize("use_duckdb", BACKENDS)
    def test_check_datapoint_reattaches_viral(self, output: str, use_duckdb: bool) -> None:
        expr = f"check_datapoint(DS_1, R {output})" if output else "check_datapoint(DS_1, R)"
        result = run(
            script=f"{_DPR}\nDS_r <- {expr};",
            data_structures={"datasets": [self._ds()]},
            datapoints={"DS_1": self._dp()},
            use_duckdb=use_duckdb,
        )
        ds_r = result["DS_r"]
        assert "VAt_1" in ds_r.components, "VAt_1 missing from result components"
        assert ds_r.components["VAt_1"].role == Role.VIRAL_ATTRIBUTE
        assert "VAt_1" in ds_r.data.columns, "VAt_1 missing from result data"
        expected = {1: "A", 2: "B"}
        for _, row in ds_r.data.iterrows():
            assert row["VAt_1"] == expected[row["Id_1"]]

    @pytest.mark.parametrize("use_duckdb", BACKENDS)
    def test_check_datapoint_executes_aggregate_rule(self, use_duckdb: bool) -> None:
        vp = "define viral propagation VP (variable VAt_1) is aggregate max end viral propagation;"
        ds = {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "Number", "role": "Viral Attribute", "nullable": True},
            ],
        }
        df = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0], "VAt_1": [100, 200, 300]})
        result = run(
            script=f"{vp}{_DPR}\nDS_r <- check_datapoint(DS_1, R all);",
            data_structures={"datasets": [ds]},
            datapoints={"DS_1": df},
            use_duckdb=use_duckdb,
        )
        # aggregate max over all datapoints -> 300 on every validation row
        assert list(result["DS_r"].data["VAt_1"]) == [300, 300, 300]


# -- Rule execution on row-preserving dataset-level operators (issue #877) --

_AGG_RULE = "define viral propagation VP (variable VAt_1) is aggregate max end viral propagation;"
_ENUM_RULE = (
    'define viral propagation VP (variable VAt_1) is when "A" then "Z"; else "D" '
    "end viral propagation;"
)


class TestViralRuleExecutionRowPreserving:
    """Row-preserving dataset-level operators must EXECUTE the rule: aggregate rules
    collapse over the whole dataset (one value on every row); enumerated rules map
    per row (issue #877)."""

    @staticmethod
    def _ds(vtype: str) -> dict:
        return {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": vtype, "role": "Viral Attribute", "nullable": True},
            ],
        }

    @pytest.mark.parametrize("expr", ["round(DS_1)", "abs(DS_1)", "DS_1 * 2"])
    @pytest.mark.parametrize("use_duckdb", BACKENDS)
    def test_aggregate_rule_is_dataset_wide(self, expr: str, use_duckdb: bool) -> None:
        result = run(
            script=f"{_AGG_RULE}\nDS_r <- {expr};",
            data_structures={"datasets": [self._ds("Number")]},
            datapoints={
                "DS_1": pd.DataFrame(
                    {"Id_1": [1, 2, 3], "Me_1": [1.0, 2.0, 3.0], "VAt_1": [100, 200, 300]}
                )
            },
            use_duckdb=use_duckdb,
        )
        # aggregate max over the whole dataset -> 300 on every result row
        assert list(result["DS_r"].data["VAt_1"]) == [300, 300, 300]

    @pytest.mark.parametrize("use_duckdb", BACKENDS)
    def test_enumerated_rule_is_per_row(self, use_duckdb: bool) -> None:
        result = run(
            script=f"{_ENUM_RULE}\nDS_r <- round(DS_1);",
            data_structures={"datasets": [self._ds("String")]},
            datapoints={
                "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": [1.0, 2.0], "VAt_1": ["A", "B"]})
            },
            use_duckdb=use_duckdb,
        )
        df = result["DS_r"].data.sort_values("Id_1")
        # "A" matches the unary clause -> "Z"; "B" is unmatched -> else "D"
        assert list(df["VAt_1"]) == ["Z", "D"]


# -- hierarchy aggregation operator --


class TestViralAttributeHierarchy:
    """hierarchy computed nodes must combine child viral values via the rule; passthrough
    rows keep their own value (issue #877)."""

    _RULE = "define viral propagation VP (variable VAt_1) is aggregate max end viral propagation;"
    _HR = (
        "define hierarchical ruleset H (valuedomain rule Id_2) is "
        "A = B + C; T = A + D end hierarchical ruleset;"
    )

    @staticmethod
    def _ds() -> dict:
        return {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "Number", "role": "Viral Attribute", "nullable": True},
            ],
        }

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

    @pytest.mark.parametrize("use_duckdb", BACKENDS)
    def test_hierarchy_combines_child_viral(self, use_duckdb: bool) -> None:
        result = run(
            script=f"{self._RULE}{self._HR}\nDS_r <- hierarchy(DS_1, H rule Id_2 non_null);",
            data_structures={"datasets": [self._ds()]},
            datapoints={"DS_1": self._dp()},
            use_duckdb=use_duckdb,
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

    _RULE = "define viral propagation VP (variable VAt_1) is aggregate max end viral propagation;"
    _HR = (
        "define hierarchical ruleset H (valuedomain rule Id_2) is "
        "A = B + C end hierarchical ruleset;"
    )

    @staticmethod
    def _ds() -> dict:
        return {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                {"name": "VAt_1", "type": "Number", "role": "Viral Attribute", "nullable": True},
            ],
        }

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

    @pytest.mark.parametrize("use_duckdb", BACKENDS)
    def test_check_hierarchy_propagates_viral(self, use_duckdb: bool) -> None:
        result = run(
            script=f"{self._RULE}{self._HR}\nDS_r <- check_hierarchy(DS_1, H rule Id_2 all);",
            data_structures={"datasets": [self._ds()]},
            datapoints={"DS_1": self._dp()},
            use_duckdb=use_duckdb,
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

    @staticmethod
    def _ds(vtype: str) -> dict:
        return {
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
                            "name": "VAt_1",
                            "type": vtype,
                            "role": "Viral Attribute",
                            "nullable": True,
                        },
                    ],
                }
            ]
        }

    @pytest.mark.parametrize("fn", ["sum", "avg"])
    @pytest.mark.parametrize("vtype", ["String", "Date", "Boolean"])
    def test_sum_avg_non_numeric_raises(self, fn: str, vtype: str) -> None:
        script = (
            f"define viral propagation VP (variable VAt_1) is aggregate {fn} "
            "end viral propagation;\nDS_r <- DS_1 * 2;"
        )
        with pytest.raises(SemanticError) as exc:
            semantic_analysis(script=script, data_structures=self._ds(vtype))
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
        semantic_analysis(script=script, data_structures=self._ds(vtype))

    @pytest.mark.parametrize("fn", ["sum", "avg"])
    def test_valuedomain_sum_avg_non_numeric_raises(self, fn: str) -> None:
        ds = {
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
        vd = {"name": "CL_X", "setlist": ["A", "B"], "type": "String"}
        script = (
            f"define viral propagation VP (valuedomain CL_X) is aggregate {fn} "
            "end viral propagation;\nDS_r <- DS_1 * 2;"
        )
        with pytest.raises(SemanticError) as exc:
            semantic_analysis(script=script, data_structures=ds, value_domains=vd)
        assert "1-3-3-5" in str(exc.value)
