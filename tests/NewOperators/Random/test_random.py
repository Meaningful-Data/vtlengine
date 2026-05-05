import warnings
from pathlib import Path

import pytest
from pytest import mark

from tests.Helper import _use_duckdb_backend
from tests.NewOperators.conftest import run_expression
from vtlengine.Exceptions import SemanticError

pytestmark = mark.input_path(Path(__file__).parent / "data")


ds_param = [
    ("1", "DS_r := DS_1[calc Me_1 := random(1, 1)];"),
    ("2", "DS_r := DS_1[calc Me_1 := random(Id_1, 1)];"),
    ("3", "DS_r := random(DS_1, 4);"),
    ("4", "DS_r := random(DS_1, 25);"),
    ("5", "DS_r := random(DS_1, 0);"),
    ("6", "DS_r := DS_1[calc Me_1 := random(Id_1, 0)];"),
    ("7", "DS_r := DS_1[calc Me_1 := random(Id_1, 0)];"),
]

error_param = [
    ("8", "DS_r := DS_1[calc Me_1 := random(Id_1, 0)];", "1-1-1-1"),
    ("9", "DS_r := random(1, -1);", "2-1-15-2"),
]


@pytest.mark.parametrize("code, expression", ds_param)
def test_case_ds(load_reference, input_paths, code, expression):
    warnings.filterwarnings("ignore", category=FutureWarning)
    result = run_expression(expression, input_paths)
    if _use_duckdb_backend():
        # DuckDB uses a different random algorithm (hash-based), so values differ.
        # Verify structure matches and values are in [0, 1).
        ref_ds = load_reference["DS_r"]
        res_ds = result["DS_r"]
        assert set(res_ds.components) == set(ref_ds.components)
        for comp_name in ref_ds.components:
            assert res_ds.components[comp_name].data_type == ref_ds.components[comp_name].data_type
            assert res_ds.components[comp_name].role == ref_ds.components[comp_name].role
        assert list(res_ds.data.columns) == list(ref_ds.data.columns)
        assert len(res_ds.data) == len(ref_ds.data)
        for col in ref_ds.data.columns:
            if ref_ds.data[col].dtype == float:
                assert (res_ds.data[col] >= 0 and res_ds.data[col] < 1).all()
    else:
        assert result == load_reference


@pytest.mark.parametrize("code, expression, error_code", error_param)
def test_errors(input_paths, code, expression, error_code):
    warnings.filterwarnings("ignore", category=FutureWarning)
    with pytest.raises(SemanticError) as context:
        run_expression(expression, input_paths)
    result = error_code == str(context.value.args[1])
    if result is False:
        print(f"\n{error_code} != {context.value.args[1]}")
    assert result
