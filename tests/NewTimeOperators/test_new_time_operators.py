import pandas as pd
import pytest

from vtlengine import DataTypes
from vtlengine.DataTypes import TimePeriod
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Scalar, Role, DataComponent, Dataset, Component
from vtlengine.Operators.Time import Month, Year

scalar_params = [
    (Month, "2022-01-23", 1),
    (Month, "2022Q1", 1),
    (Month, "2022Q4", 10),
    (Month, "2022-09-24", 9),
    (Year, "2022Q3", 2022),
    (Year, "2022-01-23", 2022),
]

dc_params = [
    (Month, pd.Series(name="TEST", data=["2022Q1", "2023-05-26"]), [1, 5]),
    (Month, pd.Series(name="TEST", data=["2022Q4", "2023-05-26"]), [10, 5]),
    (Year, pd.Series(name="TEST", data=["2022Q1", "2023-05-26"]), [2022, 2023]),
]

error_params_scalar = [(Month, "2022 / 01", "2-1-19-11"), (Year, "2022 / 01", "2-1-19-11")]

error_params_dc = [
    (Month, pd.Series(name="TEST", data=["2022 / 01", "2023-05-26"]), "2-1-19-11"),
    (Year, pd.Series(name="TEST", data=["2022 / 01", "2023-05-26"]), "2-1-19-11"),
]

month_ds_error_params = [
    (
        Month,
        pd.DataFrame(
            columns=["Id_1", "Me_1"],
            index=[0, 1, 2],
            data=[(1, "2022Q1"), (1, "2022Q2"), (1, "2022Q3")],
        ),
        "1-1-19-8",
    )
]


@pytest.mark.parametrize("op, value, reference", scalar_params)
def test_unary_time_operator_scalar(op, value, reference):
    scalar = Scalar(value, TimePeriod, value)
    result = op.evaluate(scalar)
    assert reference == result.value


@pytest.mark.parametrize("op, value, reference", dc_params)
def test_unary_time_operator_dc(op, value, reference):
    dc = DataComponent("TEST", value, TimePeriod, Role.MEASURE, False)
    result = op.evaluate(dc)
    assert all(reference == result.data.values)


@pytest.mark.parametrize("op, value, code", error_params_scalar)
def test_error_scalar(op, value, code):
    scalar = Scalar(value, TimePeriod, value)
    with pytest.raises(SemanticError, match="2-1-19-11"):
        op.evaluate(scalar)


@pytest.mark.parametrize("op, value, code", error_params_dc)
def test_error_dc(op, value, code):
    dc = DataComponent("TEST", value, TimePeriod, Role.MEASURE, False)
    with pytest.raises(SemanticError, match="2-1-19-11"):
        op.evaluate(dc)


@pytest.mark.parametrize("op, value, code", month_ds_error_params)
def test_ds_error(op, value, code):
    ds = Dataset(
        "TEST",
        components={
            "Id_1": Component(
                name="Id_1", data_type=DataTypes.Integer, role=Role.IDENTIFIER, nullable=False
            ),
            "Me_1": Component(
                name="Me_1", data_type=DataTypes.Date, role=Role.MEASURE, nullable=False
            ),
        },
        data=value,
    )
    with pytest.raises(SemanticError, match="1-1-19-8"):
        op.evaluate(ds)
