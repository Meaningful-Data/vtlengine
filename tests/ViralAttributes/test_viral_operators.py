"""Tests for viral attribute propagation through all operator categories."""

import pandas as pd
import pytest

from vtlengine import run
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
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_aggr_clause_preserves_viral_attrs(self, expr: str, num_viral: int) -> None:
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
    corrupted the result (component/data mismatch -> downstream crash)."""

    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_if_dataset_condition_preserves_viral_attrs(self, num_viral: int) -> None:
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
        )
        _assert_viral_attrs(result, num_viral)
        for va_name in VA_NAMES[:num_viral]:
            assert va_name in result["DS_r"].data.columns, f"{va_name} missing from result data"

    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_count_over_if_dataset_condition(self, num_viral: int) -> None:
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
        )
        _assert_viral_attrs(result, num_viral)
        for va_name in VA_NAMES[:num_viral]:
            assert va_name in result["DS_r"].data.columns, f"{va_name} missing from result data"

    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_case_dataset_condition_preserves_viral_attrs(self, num_viral: int) -> None:
        """A dataset-level ``case`` must not leave phantom suffixed viral columns."""
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
        )
        _assert_viral_attrs(result, num_viral)
        _assert_component_data_parity(result)

    @pytest.mark.parametrize(
        "expr",
        ["nvl(DS_1, DS_2)", "nvl(DS_1, 0.0)"],
    )
    @pytest.mark.parametrize("num_viral", [1, 2, 3])
    def test_nvl_preserves_viral_attrs(self, expr: str, num_viral: int) -> None:
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
