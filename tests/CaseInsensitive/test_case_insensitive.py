"""Tests for case-insensitive regular name resolution (VTL 2.1 spec)."""

import pandas as pd
import pytest

from vtlengine import run
from vtlengine.Exceptions import SemanticError

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BASE_STRUCTURES = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}

BASE_DATAPOINTS = {
    "DS_1": pd.DataFrame({"Id_1": [1, 1, 1], "Id_2": ["A", "B", "C"], "Me_1": [10.0, 20.0, 30.0]})
}

HR_DATAPOINTS = {
    "DS_1": pd.DataFrame(
        {"Id_1": [1, 1, 1, 1], "Id_2": ["A", "B", "C", "D"], "Me_1": [10.0, 20.0, 30.0, None]}
    )
}

HR_RULE_BODY = """\
    E = A + B errorcode "e1" errorlevel 1
end hierarchical ruleset"""

DPR_RULE_BODY = """\
    when Id_2 = "A" then Me_1 >= 0 errorcode "err1"
end datapoint ruleset"""


def _run(script: str, datapoints: dict = BASE_DATAPOINTS) -> dict:
    return run(script=script, data_structures=BASE_STRUCTURES, datapoints=datapoints)


# ---------------------------------------------------------------------------
# 1. Dataset name resolution
# ---------------------------------------------------------------------------

dataset_name_params = [
    pytest.param("ds_1", id="all_lower"),
    pytest.param("Ds_1", id="mixed_1"),
    pytest.param("DS_1", id="original"),
    pytest.param("dS_1", id="mixed_2"),
]


@pytest.mark.parametrize("alias", dataset_name_params)
def test_dataset_name_case_variants(alias):
    result = _run(f"DS_r <- {alias};")
    assert "DS_r" in result
    assert list(result["DS_r"].data.columns) == ["Id_1", "Id_2", "Me_1"]


def test_dataset_preserves_original_name():
    result = _run("My_Result <- ds_1;")
    assert "My_Result" in result


def test_dataset_chained_resolution():
    result = _run("DS_r <- DS_1; DS_r2 <- ds_r;")
    assert "DS_r2" in result
    pd.testing.assert_frame_equal(result["DS_r"].data, result["DS_r2"].data)


# ---------------------------------------------------------------------------
# 2. Duplicate assignment detection
# ---------------------------------------------------------------------------

duplicate_params = [
    pytest.param("DS_r <- DS_1; DS_R <- DS_1;", id="different_case"),
    pytest.param("DS_r <- DS_1; DS_r <- DS_1;", id="same_case"),
    pytest.param("DS_r <- DS_1; ds_r <- DS_1;", id="all_lower"),
]


@pytest.mark.parametrize("script", duplicate_params)
def test_duplicate_assignment_raises(script):
    with pytest.raises(SemanticError, match="1-2-2"):
        _run(script)


# ---------------------------------------------------------------------------
# 3. Component name resolution (calc, filter, rename)
# ---------------------------------------------------------------------------

component_calc_params = [
    pytest.param(
        "DS_r <- ds_1[calc me_2 := me_1 * 2];",
        ["me_2"],
        id="calc_lowercase",
    ),
    pytest.param(
        "DS_r <- ds_1[calc me_2 := Me_1, mE_3 := ME_1 + me_1];",
        ["me_2", "mE_3"],
        id="calc_mixed_case",
    ),
]


@pytest.mark.parametrize("script, expected_comps", component_calc_params)
def test_calc_case_insensitive(script, expected_comps):
    result = _run(script)
    for comp in expected_comps:
        assert comp in result["DS_r"].components


calc_override_params = [
    pytest.param(
        "DS_r <- DS_1[calc ME_1 := Me_1 * 2];",
        ["Id_1", "Id_2", "ME_1"],
        id="calc_override_upper",
    ),
    pytest.param(
        "DS_r <- DS_1[calc me_1 := Me_1 * 2];",
        ["Id_1", "Id_2", "me_1"],
        id="calc_override_lower",
    ),
]


@pytest.mark.parametrize("script, expected_cols", calc_override_params)
def test_calc_output_columns_case_insensitive(script, expected_cols):
    """Output DataFrame columns must match component names (no duplicates)."""
    result = _run(script)
    ds = result["DS_r"]
    assert list(ds.components.keys()) == expected_cols
    assert list(ds.data.columns) == expected_cols


filter_params = [
    pytest.param("DS_r <- ds_1[filter me_1 > 15];", 2, id="lowercase_measure"),
    pytest.param("DS_r <- ds_1[filter ME_1 > 15];", 2, id="uppercase_measure"),
    pytest.param("DS_r <- ds_1[filter Me_1 > 25];", 1, id="original_case"),
]


@pytest.mark.parametrize("script, expected_rows", filter_params)
def test_filter_case_insensitive(script, expected_rows):
    result = _run(script)
    assert len(result["DS_r"].data) == expected_rows


rename_params = [
    pytest.param("me_1", "Me_New", id="lowercase_old"),
    pytest.param("ME_1", "Me_New", id="uppercase_old"),
    pytest.param("Me_1", "Me_Renamed", id="original_case_old"),
]


@pytest.mark.parametrize("old_name, new_name", rename_params)
def test_rename_case_insensitive(old_name, new_name):
    result = _run(f"DS_r <- ds_1[rename {old_name} to {new_name}];")
    assert new_name in result["DS_r"].components
    assert "Me_1" not in result["DS_r"].components


# ---------------------------------------------------------------------------
# 4. Hierarchical ruleset name resolution
# ---------------------------------------------------------------------------

hr_params = [
    pytest.param("hr1", "HR1", "Id_2", "Id_2", id="name_upper"),
    pytest.param("hr1", "hr1", "Id_2", "id_2", id="comp_lower"),
    pytest.param("My_HR", "MY_HR", "Id_2", "ID_2", id="both_different"),
    pytest.param("hr1", "Hr1", "id_2", "ID_2", id="all_mixed"),
]


@pytest.mark.parametrize("def_name, call_name, def_comp, call_comp", hr_params)
def test_hr_case_insensitive(def_name, call_name, def_comp, call_comp):
    script = f"""
        define hierarchical ruleset {def_name} (variable rule {def_comp}) is
            {HR_RULE_BODY};
        DS_r <- hierarchy(DS_1, {call_name} rule {call_comp} computed);
    """
    result = _run(script, datapoints=HR_DATAPOINTS)
    assert "DS_r" in result
    assert "Id_2" in result["DS_r"].components


# ---------------------------------------------------------------------------
# 5. Datapoint ruleset name resolution
# ---------------------------------------------------------------------------

dpr_params = [
    pytest.param("dpr1", "DPR1", "Id_2, Me_1", "Id_2, Me_1", id="name_upper"),
    pytest.param("dpr1", "dpr1", "ID_2, ME_1", "id_2, me_1", id="comps_swapped"),
    pytest.param("My_DPR", "MY_DPR", "ID_2, ME_1", "id_2, me_1", id="both_different"),
]


@pytest.mark.parametrize("def_name, call_name, def_comps, call_comps", dpr_params)
def test_dpr_case_insensitive(def_name, call_name, def_comps, call_comps):
    script = f"""
        define datapoint ruleset {def_name} (variable {def_comps}) is
            {DPR_RULE_BODY};
        DS_r := check_datapoint(DS_1, {call_name} components {call_comps} invalid);
    """
    result = _run(script)
    assert result is not None


# ---------------------------------------------------------------------------
# 6. UDO name resolution
# ---------------------------------------------------------------------------

udo_params = [
    pytest.param(
        """
        define operator my_op (ds dataset) returns dataset is ds end operator;
        DS_r <- MY_OP(DS_1);
        """,
        "DS_r",
        id="simple_upper",
    ),
    pytest.param(
        """
        define operator my_op (ds dataset) returns dataset is ds end operator;
        DS_r <- My_Op(ds_1);
        """,
        "DS_r",
        id="simple_mixed",
    ),
    pytest.param(
        """
        define operator suma (ds1 dataset, ds2 dataset) returns dataset is ds1 + ds2 end operator;
        define operator drop_id (ds dataset, comp component)
            returns dataset is max(ds group except comp) end operator;
        DS_r <- DROP_ID(SUMA(ds_1, Ds_1), Id_2);
        """,
        "DS_r",
        id="nested_mixed",
    ),
]


@pytest.mark.parametrize("script, expected_ds", udo_params)
def test_udo_case_insensitive(script, expected_ds):
    result = _run(script)
    assert expected_ds in result


def test_udo_duplicate_definition_different_case():
    script = """
        define operator my_op (ds dataset) returns dataset is ds end operator;
        define operator MY_OP (ds dataset) returns dataset is ds end operator;
        DS_r <- my_op(DS_1);
    """
    with pytest.raises((ValueError, SemanticError)):
        _run(script)


# ---------------------------------------------------------------------------
# 7. Aggregation with case-insensitive component refs
# ---------------------------------------------------------------------------

agg_params = [
    pytest.param("sum(ds_1 group by id_1)", ["Id_1", "Me_1"], id="group_by_lower"),
    pytest.param("sum(DS_1 group by Id_1)", ["Id_1", "Me_1"], id="group_by_original"),
    pytest.param("max(ds_1 group except id_2)", ["Id_1", "Me_1"], id="group_except_lower"),
    pytest.param("max(ds_1 group except ID_2)", ["Id_1", "Me_1"], id="group_except_upper"),
    pytest.param("sum(ds_1 group by ID_1)", ["Id_1", "Me_1"], id="group_by_upper"),
]


@pytest.mark.parametrize("expr, expected_comps", agg_params)
def test_aggregation_case_insensitive(expr, expected_comps):
    result = _run(f"DS_r <- {expr};")
    assert "DS_r" in result
    for comp in expected_comps:
        assert comp in result["DS_r"].components


aggr_override_params = [
    pytest.param(
        "DS_r <- DS_1[aggr ME_1 := sum(Me_1) group by Id_1];",
        ["Id_1", "ME_1"],
        id="aggr_override_upper",
    ),
    pytest.param(
        "DS_r <- DS_1[aggr me_1 := sum(Me_1) group by Id_1];",
        ["Id_1", "me_1"],
        id="aggr_override_lower",
    ),
]


@pytest.mark.parametrize("script, expected_cols", aggr_override_params)
def test_aggr_output_columns_case_insensitive(script, expected_cols):
    """Output DataFrame columns must match component names (no duplicates)."""
    result = _run(script)
    ds = result["DS_r"]
    assert list(ds.components.keys()) == expected_cols
    assert list(ds.data.columns) == expected_cols


# ---------------------------------------------------------------------------
# 8. Scalar name resolution
# ---------------------------------------------------------------------------

scalar_params = [
    pytest.param("my_sc", "my_sc", 43, id="lower"),
    pytest.param("My_Sc", "my_sc", 43, id="mixed"),
    pytest.param("MY_SC", "my_sc", 43, id="upper"),
]


@pytest.mark.parametrize("def_name, ref_name, expected_value", scalar_params)
def test_scalar_case_insensitive(def_name, ref_name, expected_value):
    script = f"{def_name} <- 42; DS_r <- {ref_name} + 1;"
    result = run(script=script, data_structures={"datasets": []}, datapoints={})
    assert "DS_r" in result
    assert result["DS_r"].value == expected_value


# ---------------------------------------------------------------------------
# 9. End-to-end mixed operations
# ---------------------------------------------------------------------------

e2e_params = [
    pytest.param(
        "DS_r <- DS_1; DS_r2 <- ds_r;",
        ["DS_r", "DS_r2"],
        id="chained_datasets",
    ),
    pytest.param(
        "DS_r <- ds_1[calc me_2 := Me_1, mE_3 := ME_1 + me_1];",
        ["DS_r"],
        id="calc_mixed_refs",
    ),
    pytest.param(
        "DS_r <- DS_1; DS_r2 <- ds_r; DS_r3 <- ds_1[calc me_2 := Me_1];",
        ["DS_r", "DS_r2", "DS_r3"],
        id="full_pipeline",
    ),
]


@pytest.mark.parametrize("script, expected_datasets", e2e_params)
def test_end_to_end(script, expected_datasets):
    result = _run(script)
    for ds in expected_datasets:
        assert ds in result
