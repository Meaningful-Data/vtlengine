import pandas as pd
import pytest

from vtlengine.DataTypes import TimePeriod
from vtlengine.Model import Scalar, Role, DataComponent
from vtlengine.Operators.Time import Month

month_scalar_params = [
    (Month, '2022-01-23', 1),
    (Month, '2022Q1', 1),
    (Month, '2022Q4', 10),
    (Month, '2022-09-24', 9)
]

month_dc_params = [
    (Month, pd.Series(name='TEST', data=['2022Q1', '2023-05-26']), [1, 5]),
    (Month, pd.Series(name='TEST', data=['2022Q4', '2023-05-26']), [10, 5])
]

error_params_scalar = [
    (Month, '2022Q5', KeyError)
]

error_params_dc = [
    (Month, pd.Series(name='TEST', data=['2022Q5', '2023-05-26']), [10, 5])
]


@pytest.mark.parametrize("op, value, reference", month_scalar_params)
def test_month_scalar(op, value, reference):
    scalar = Scalar(value, TimePeriod, value)
    result = op.evaluate(scalar)
    assert reference == result.value


@pytest.mark.parametrize("op, value, reference", month_dc_params)
def test_month_dc(op, value, reference):
    dc = DataComponent('TEST', value, TimePeriod, Role.MEASURE, False)
    result = op.evaluate(dc)
    assert all(reference == result.data.values)


@pytest.mark.parametrize('op, value, error', error_params_scalar)
def test_month_error_scalar(op, value, error):
    scalar = Scalar(value, TimePeriod, value)
    with pytest.raises(KeyError, match=None):
        op.evaluate(scalar)


@pytest.mark.parametrize('op, value, error', error_params_dc)
def test_month_error_dc(op, value, error):
    dc = DataComponent('TEST', value, TimePeriod, Role.MEASURE, False)
    with pytest.raises(KeyError, match=None):
        op.evaluate(dc)
