import warnings
from pathlib import Path

import pytest
from pytest import mark

from tests.NewOperators.conftest import run_expression, run_scalar_expression
from vtlengine.Exceptions import RunTimeError, SemanticError

pytestmark = mark.input_path(Path(__file__).parent / "data")


ds_param = [
    ("1", "DS_r := string_distance(levenshtein, DS_1, DS_2);"),
    ("2", "DS_r := string_distance(damerau_levenshtein, DS_1, DS_2);"),
    (
        "3",
        'DS_r := DS_1[calc Me_2 := string_distance(hamming, Me_1, "fob")];',
    ),
    ("4", "DS_r := string_distance(jaro_winkler, DS_1, DS_2);"),
    ("5", "DS_r := string_distance(hamming, DS_1, DS_2);"),
]


@pytest.mark.parametrize("code, expression", ds_param)
def test_string_distance_ds(load_reference, input_paths, code, expression):
    warnings.filterwarnings("ignore", category=FutureWarning)
    result = run_expression(expression, input_paths)
    assert result == load_reference


def test_string_distance_scalar():
    """Scalar form: string_distance(levenshtein, 'foo', 'fo') => 1."""
    result = run_scalar_expression('DS_r := string_distance(levenshtein, "foo", "fo");')
    assert result["DS_r"].value == 1


def test_hamming_length_mismatch():
    """Hamming on strings of unequal length must raise."""
    expression = 'DS_r := string_distance(hamming, "foo", "fooo");'
    with pytest.raises((SemanticError, RunTimeError)):
        run_scalar_expression(expression)
