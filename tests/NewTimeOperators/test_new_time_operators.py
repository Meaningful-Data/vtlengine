import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from vtlengine import DataTypes
from vtlengine.API import create_ast
from vtlengine.DataTypes import Date, Integer, String, TimeInterval
from vtlengine.Exceptions import SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer
from vtlengine.Model import Scalar, Role, Dataset, Component
from vtlengine.Operators.Time import Month

ds_1 = Dataset(
    name="DS_1",
    components={
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=Date, role=Role.MEASURE, nullable=True),
    },
    data=pd.DataFrame(
        columns=["Id_1", "Me_1"],
        index=[0, 1, 2],
        data=[("A", "2022-01-30"), ("B", "2022-08-21"), ("C", None)],
    ),
)
ds_2 = Dataset(
    name="DS_1",
    components={
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=String, role=Role.MEASURE, nullable=True),
    },
    data=pd.DataFrame(
        columns=["Id_1", "Me_1"],
        index=[0, 1, 2],
        data=[("A", "2022-01-30"), ("B", "2022-08-21"), ("C", None)],
    ),
)
ds_3 = Dataset(
    name="DS_1",
    components={
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=TimeInterval, role=Role.MEASURE, nullable=True),
    },
    data=pd.DataFrame(
        columns=["Id_1", "Me_1"],
        index=[0, 1, 2],
        data=[("A", "2022-01-30"), ("B", "2022-08-21"), ("C", None)],
    ),
)

ds_4 = Dataset(
    name="DS_1",
    components={
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=Date, role=Role.MEASURE, nullable=True),
    },
    data=pd.DataFrame(
        columns=["Id_1", "Me_1"],
        index=[0, 1],
        data=[("A", "782"), ("B", None)],
    ),
)
ds_slash_error = Dataset(
    name="DS_1",
    components={
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=Date, role=Role.MEASURE, nullable=True),
    },
    data=pd.DataFrame(
        columns=["Id_1", "Me_1"],
        index=[0, 1, 2],
        data=[("A", "2022-01-30"), ("B", "2022-08-21/2023-09-21"), ("C", None)],
    ),
)
ds_error_params = [
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

vtl_expression_test_params = [
    ('year(cast("2023-01-12", date))', 2023),
    ('year(cast("2022Q1", time_period))', 2022),
    ("DS_1[calc Me_2 := year(Me_1)]", pd.Series(name="Me_2", data=[2022, 2022, None])),
    ('month(cast("2023-01-12", date))', 1),
    ('month(cast("2022Q1", time_period))', 1),
    ("DS_1[calc Me_2 := month(Me_1)]", pd.Series(name="Me_2", data=[1, 8, None])),
    ('dayofmonth(cast("2023-01-12", date))', 12),
    ('dayofmonth(cast("2022Q1", time_period))', 31),
    ("DS_1[calc Me_2 := dayofmonth(Me_1)]", pd.Series(name="Me_2", data=[30, 21, None])),
    ('dayofyear(cast("2023-01-12", date))', 12),
    ('dayofyear(cast("2022Q1", time_period))', 90),
    ("DS_1[calc Me_2 := dayofyear(Me_1)]", pd.Series(name="Me_2", data=[30, 233, None])),
]
me_str_params = [("DS_1[calc Me_2 := year(Me_1)]", "1-1-19-10")]

me_time_interval_params = [("DS_1[calc Me_2 := year(Me_1)]", "1-1-19-10")]

slash_in_vtl_expression_error = [
    ("DS_1[calc Me_2 := year(Me_1)]", "2-1-19-11"),
    ("DS_1[calc Me_2 := month(Me_1)]", "2-1-19-11"),
    ("DS_1[calc Me_2 := dayofyear(Me_1)]", "2-1-19-11"),
    ("DS_1[calc Me_2 := dayofmonth(Me_1)]", "2-1-19-11"),
]

transfomration_with_masks_params = [
    ("DS_1[calc Me_2 := daytoyear(Me_1)]", pd.Series(name="Me_2", data=["P2Y52D", None]))
]


@pytest.mark.parametrize("op, value, code", ds_error_params)
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


@pytest.mark.parametrize("text, reference", vtl_expression_test_params)
def test_vtl_expression_unary_time_op(text, reference):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({"DS_1": ds_1})
    result = interpreter.visit(ast)
    if isinstance(result["DS_r"], Scalar):
        assert result["DS_r"].value == reference
        assert result["DS_r"].data_type == Integer
    else:
        assert_series_equal(result["DS_r"].data["Me_2"], reference)


@pytest.mark.parametrize("text, reference", transfomration_with_masks_params)
def test_vtl_expression_unary_time_op_with_masks(text, reference):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({"DS_1": ds_4})
    result = interpreter.visit(ast)
    if isinstance(result["DS_r"], Scalar):
        assert result["DS_r"].value == reference
        assert result["DS_r"].data_type == Integer
    else:
        assert_series_equal(result["DS_r"].data["Me_2"], reference)


@pytest.mark.parametrize("text, code", me_str_params)
def test_vtl_expression_unary_time_op_me_str(text, code):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({"DS_1": ds_2})
    with pytest.raises(SemanticError, match="1-1-19-10"):
        interpreter.visit(ast)


@pytest.mark.parametrize("text, reference", me_time_interval_params)
def test_vtl_expression_unary_time_op_me_time_interval(text, reference):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({"DS_1": ds_2})
    with pytest.raises(SemanticError, match="1-1-19-10"):
        interpreter.visit(ast)


@pytest.mark.parametrize("text, code", slash_in_vtl_expression_error)
def test_slash_in_date_error(text, code):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({"DS_1": ds_slash_error})
    with pytest.raises(SemanticError, match="2-1-19-11"):
        interpreter.visit(ast)
