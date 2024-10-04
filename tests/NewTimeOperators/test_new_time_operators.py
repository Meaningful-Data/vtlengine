import pytest

from vtlengine.DataTypes import TimePeriod
from vtlengine.Model import Scalar
from vtlengine.Operators.Time import Year

year_params = [
    (Year, '2022Q1', 2022),
    (Year, '2022-01-23', 2022)
]


@pytest.mark.parametrize("op, value, reference", year_params)
def test_year_op(op, value, reference):
    scalar = Scalar(value, TimePeriod, value)
    result = op.evaluate(scalar)
    assert reference == result.value
