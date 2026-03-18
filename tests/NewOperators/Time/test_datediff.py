import warnings
from pathlib import Path

import pytest

from tests.NewOperators.conftest import _build_run_inputs, use_duckdb
from vtlengine.API import run
from vtlengine.DataTypes import Integer
from vtlengine.Exceptions import SemanticError

base_path = Path(__file__).parent / "data"
pytestmark = pytest.mark.input_path(base_path)

ds_param = [
    ("21", "DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];"),
    ("22", "DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];"),
    ("23", "DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];"),
    ("24", "DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];"),
]

error_param = [
    ("25", "DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];", "1-1-1-2"),
    ("26", "DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];", "1-1-1-2"),
]

scalar_time_params = [
    ('datediff(cast("2020-12-14", date),cast("2021-04-20", date))', 127),
    ('datediff(cast("2020-01-01",date),cast("2021-01-01",date))', 366),
    ('datediff(cast("2022Q1",time_period),cast("2023Q2",time_period))', 456),
    ('datediff(cast("2020D1",time_period),cast("2020D15",time_period))', 14),
]

scalar_time_error_params = [
    ('datediff(cast("2022Q1",date),cast("2023Q2",time_period))', SemanticError, "1-1-1-2"),
    ('datediff(cast("2020D1",time_period),cast("2020D15",date))', SemanticError, "1-1-1-2"),
    ('datediff(cast("2022-06-30",date),cast("2023Q2",time_period))', SemanticError, "1-1-1-2"),
    ('datediff(cast("2022Q2",time_period),cast("2023-06-30",date))', SemanticError, "1-1-1-2"),
]


@pytest.mark.parametrize("code, expression", ds_param)
def test_case_ds(load_reference, code, expression):
    warnings.filterwarnings("ignore", category=FutureWarning)
    data_structures, datapoints = _build_run_inputs(code, base_path)
    result = run(
        script=expression,
        data_structures=data_structures,
        datapoints=datapoints,
        return_only_persistent=False,
        use_duckdb=use_duckdb,
    )
    assert result == load_reference


@pytest.mark.parametrize("text, reference", scalar_time_params)
def test_unary_time_scalar(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    result = run(
        script=expression,
        data_structures=[],
        datapoints=[],
        return_only_persistent=False,
        use_duckdb=use_duckdb,
    )
    assert result["DS_r"].value == reference
    assert result["DS_r"].data_type == Integer


@pytest.mark.parametrize("code, expression, error_code", error_param)
def test_errors(code, expression, error_code):
    warnings.filterwarnings("ignore", category=FutureWarning)
    data_structures, datapoints = _build_run_inputs(code, base_path)
    with pytest.raises(SemanticError) as context:
        run(
            script=expression,
            data_structures=data_structures,
            datapoints=datapoints,
            return_only_persistent=False,
        )
    result = error_code == str(context.value.args[1])
    if result is False:
        print(f"\n{error_code} != {context.value.args[1]}")
    assert result


@pytest.mark.parametrize("text, exception_type, exception_message", scalar_time_error_params)
def test_errors_time_scalar(text, exception_type, exception_message):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    with pytest.raises(exception_type, match=f".*{exception_message}"):
        run(
            script=expression,
            data_structures=[],
            datapoints=[],
            return_only_persistent=False,
            use_duckdb=use_duckdb,
        )
