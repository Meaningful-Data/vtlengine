import warnings
from pathlib import Path

import pytest

from tests.NewOperators.conftest import _build_run_inputs
from vtlengine.API import run
from vtlengine.Exceptions import SemanticError

base_path = Path(__file__).parent / "data"
pytestmark = pytest.mark.input_path(base_path)

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
def test_random(load_reference, code, expression):
    warnings.filterwarnings("ignore", category=FutureWarning)
    data_structures, datapoints = _build_run_inputs(code, base_path)
    result = run(
        script=expression,
        data_structures=data_structures,
        datapoints=datapoints,
        return_only_persistent=False,
    )
    assert result == load_reference


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
