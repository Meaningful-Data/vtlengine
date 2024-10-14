import pandas as pd
import pytest

from vtlengine.DataTypes import TimePeriod
from vtlengine.Model import Scalar, DataComponent, Role
from vtlengine.Operators.Time import Year

year_params_scalar = [
    (Year, '2022Q1', 2022),
    (Year, '2022-01-23', 2022)
]

year_params_dc = [
    (Year, pd.Series(name='TEST', data=['2022Q1', '2023Q2', '2021Q3']), [2022, 2023, 2021])
]


@pytest.mark.parametrize("op, value, reference", year_params_scalar)
def test_year_op(op, value, reference):
    scalar = Scalar(value, TimePeriod, value)
    result = op.evaluate(scalar)
    assert reference == result.value


@pytest.mark.parametrize("op, value, reference", year_params_dc)
def test_year_dc(op, value, reference):
    dc = DataComponent('TEST', value, TimePeriod, Role.MEASURE, False)
    result = op.evaluate(dc)
    assert all(reference == result.data.values)
