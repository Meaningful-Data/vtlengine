import warnings
from pathlib import Path

import pytest
from pytest import mark

from tests.NewOperators.conftest import run_expression, run_scalar_expression
from vtlengine.Exceptions import SemanticError

pytestmark = mark.input_path(Path(__file__).parent / "data")


ds_param = [
    pytest.param("1", "DS_r := string_distance(levenshtein, DS_1, DS_2);", id="levenshtein"),
    pytest.param(
        "2",
        "DS_r := string_distance(damerau_levenshtein, DS_1, DS_2);",
        id="damerau_levenshtein",
    ),
    pytest.param(
        "3",
        'DS_r := DS_1[calc Me_2 := string_distance(hamming, Me_1, "fob")];',
        id="hamming_calc",
    ),
    pytest.param("4", "DS_r := string_distance(jaro_winkler, DS_1, DS_2);", id="jaro_winkler"),
    pytest.param("5", "DS_r := string_distance(hamming, DS_1, DS_2);", id="hamming_ds_ds"),
]


@pytest.mark.parametrize("code, expression", ds_param)
def test_string_distance_ds(load_reference, input_paths, code, expression):
    warnings.filterwarnings("ignore", category=FutureWarning)
    result = run_expression(expression, input_paths)
    assert result == load_reference


def test_string_distance_scalar():
    result = run_scalar_expression('DS_r := string_distance(levenshtein, "foo", "fo");')
    assert result["DS_r"].value == 1


def test_hamming_length_mismatch():
    expression = 'DS_r := string_distance(hamming, "foo", "fooo");'
    with pytest.raises(SemanticError) as exc:
        run_scalar_expression(expression)
    assert exc.value.args[1] == "1-1-18-11"
