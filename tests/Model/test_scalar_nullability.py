import json

import pandas as pd
import pytest

from vtlengine import run, semantic_analysis
from vtlengine.API._InternalApi import _load_dataset_from_structure
from vtlengine.AST import Constant
from vtlengine.DataTypes import Boolean, Integer, Number
from vtlengine.DataTypes import Date as _DateType
from vtlengine.Interpreter import InterpreterAnalyzer
from vtlengine.Model import Role, Scalar
from vtlengine.Operators.CastOperator import Cast
from vtlengine.Operators.Comparison import Between, IsNull
from vtlengine.Operators.Conditional import Case, Nvl
from vtlengine.Operators.Numeric import BinPlus, UnMinus
from vtlengine.Operators.RoleSetter import Measure
from vtlengine.Operators.Time import Time_Aggregation


def _scalar(value, nullable, data_type=Integer):
    return Scalar(name="x", data_type=data_type, value=value, nullable=nullable)


def _interp():
    return InterpreterAnalyzer(datasets={})


# --- Model: construction defaults -------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, True),  # default
        ({"nullable": True}, True),
        ({"nullable": False}, False),
    ],
)
def test_scalar_init_nullable(kwargs, expected):
    s = Scalar(name="s", data_type=Integer, value=None, **kwargs)
    assert s.nullable is expected


# --- Model: serialization ---------------------------------------------------


@pytest.mark.parametrize("nullable", [True, False])
def test_scalar_to_dict_includes_nullable(nullable):
    s = Scalar(name="s", data_type=Integer, value=5, nullable=nullable)
    assert s.to_dict()["nullable"] is nullable


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"name": "s", "type": "Integer", "value": 5, "nullable": False}, False),
        ({"name": "s", "type": "Integer", "value": 5, "nullable": True}, True),
        ({"name": "s", "type": "Integer", "value": 5}, True),  # missing key -> default True
    ],
)
def test_scalar_from_json_nullable(payload, expected):
    s = Scalar.from_json(json.dumps(payload))
    assert s.nullable is expected


def test_scalar_eq_ignores_nullable():
    a = Scalar(name="s", data_type=Integer, value=5, nullable=True)
    b = Scalar(name="s", data_type=Integer, value=5, nullable=False)
    assert a == b


# --- Interpreter: constants -------------------------------------------------


@pytest.mark.parametrize(
    ("type_", "value", "expected"),
    [
        ("INTEGER_CONSTANT", 5, False),
        ("STRING_CONSTANT", "a", False),
        ("BOOLEAN_CONSTANT", True, False),
        ("NULL_CONSTANT", None, True),
    ],
)
def test_visit_constant_nullable(type_, value, expected):
    node = Constant(
        line_start=0, column_start=0, line_stop=0, column_stop=0, type_=type_, value=value
    )
    result = _interp().visit_Constant(node)
    assert result.nullable is expected


# --- Base operator scalar validation ----------------------------------------


@pytest.mark.parametrize(
    ("left_nullable", "right_nullable", "expected"),
    [
        (True, False, True),
        (False, True, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_binary_scalar_validation_nullable(left_nullable, right_nullable, expected):
    result = BinPlus.scalar_validation(_scalar(1, left_nullable), _scalar(2, right_nullable))
    assert result.nullable is expected


@pytest.mark.parametrize("nullable", [True, False])
def test_unary_scalar_validation_propagates_nullable(nullable):
    result = UnMinus.scalar_validation(_scalar(1, nullable))
    assert result.nullable is nullable


# --- RoleSetter (root-cause) ------------------------------------------------


@pytest.mark.parametrize(
    ("value", "nullable", "expected"),
    [
        (None, True, True),  # scalar input semantics
        (5, False, False),  # constant semantics
        (5, True, True),
        (None, False, False),
    ],
)
def test_rolesetter_uses_scalar_nullable(value, nullable, expected):
    result = Measure.validate(Scalar(name="m", data_type=Integer, value=value, nullable=nullable))
    assert result.role == Role.MEASURE
    assert result.nullable is expected


# --- Comparison -------------------------------------------------------------


@pytest.mark.parametrize("operand_nullable", [True, False])
def test_isnull_scalar_is_not_nullable(operand_nullable):
    result = IsNull.scalar_validation(_scalar(None, operand_nullable))
    assert result.data_type == Boolean
    assert result.nullable is False


@pytest.mark.parametrize(
    ("op_n", "lo_n", "hi_n", "expected"),
    [
        (True, False, False, True),
        (False, True, False, True),
        (False, False, True, True),
        (False, False, False, False),
    ],
)
def test_between_scalar_propagates_nullable(op_n, lo_n, hi_n, expected):
    result = Between.validate(
        _scalar(5, op_n),
        Scalar(name="lo", data_type=Integer, value=1, nullable=lo_n),
        Scalar(name="hi", data_type=Integer, value=10, nullable=hi_n),
    )
    assert result.nullable is expected


# --- Conditional ------------------------------------------------------------


@pytest.mark.parametrize(
    ("left_nullable", "right_nullable"),
    [(True, False), (True, True), (False, False)],
)
def test_nvl_scalar_is_not_nullable(left_nullable, right_nullable):
    left = Scalar(name="l", data_type=Integer, value=None, nullable=left_nullable)
    right = Scalar(name="r", data_type=Integer, value=0, nullable=right_nullable)
    result = Nvl.validate(left, right)
    assert result.nullable is False


@pytest.mark.parametrize(
    ("then_nullable", "else_nullable", "expected"),
    [
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_case_scalar_propagates_nullable(then_nullable, else_nullable, expected):
    cond = Scalar(name="c", data_type=Boolean, value=True, nullable=False)
    then_op = Scalar(name="t", data_type=Integer, value=1, nullable=then_nullable)
    else_op = Scalar(name="e", data_type=Integer, value=2, nullable=else_nullable)
    result = Case.validate([cond], [then_op], else_op)
    assert result.nullable is expected


# --- Cast -------------------------------------------------------------------


@pytest.mark.parametrize("nullable", [True, False])
def test_cast_scalar_validation_carries_nullable(nullable):
    result = Cast.scalar_validation(_scalar(None, nullable), Number)
    assert result.data_type == Number
    assert result.nullable is nullable


@pytest.mark.parametrize("nullable", [True, False])
def test_cast_scalar_value_carries_nullable(nullable):
    result = Cast.cast_scalar(_scalar(5, nullable), Number)
    assert result.nullable is nullable


# --- Time -------------------------------------------------------------------


@pytest.mark.parametrize("nullable", [True, False])
def test_time_scalar_validation_carries_nullable(nullable):
    operand = Scalar(name="x", data_type=_DateType, value="2020-01-01", nullable=nullable)
    result = Time_Aggregation.scalar_validation(operand, None, "M", "first")
    assert result.nullable is nullable


# --- Scalar input parsing ---------------------------------------------------


@pytest.mark.parametrize(
    ("scalar_json", "expected"),
    [
        ({"name": "sc_1", "type": "Integer"}, True),  # default
        ({"name": "sc_1", "type": "Integer", "nullable": True}, True),
        ({"name": "sc_1", "type": "Integer", "nullable": False}, False),
    ],
)
def test_scalar_input_nullable(scalar_json, expected):
    _datasets, scalars = _load_dataset_from_structure({"scalars": [scalar_json]})
    assert scalars["sc_1"].nullable is expected


# --- End-to-end: semantic_analysis() and run() must agree -------------------


@pytest.mark.parametrize(
    ("calc_expr", "expected"),
    [
        ("5", False),  # constant -> non-nullable
        ("m1 + 5", True),  # nullable measure propagates -> nullable
    ],
)
def test_calc_measure_structure_consistent(calc_expr, expected):
    # Use persistent assignment (<-) so run() returns DS_r with default return_only_persistent=True.
    # In VTL, <- is the persistent (PUT_SYMBOL) assignment; := is the temporary assignment.
    script = f"DS_r <- DS_1[calc m2 := {calc_expr}];"
    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "m1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    datapoints = {"DS_1": pd.DataFrame({"Id_1": [1, 2], "m1": [10.0, 20.0]})}

    sem = semantic_analysis(script=script, data_structures=data_structures)
    runres = run(script=script, data_structures=data_structures, datapoints=datapoints)

    sem_m2 = sem["DS_r"].components["m2"].nullable
    run_m2 = runres["DS_r"].components["m2"].nullable
    # Both API entry points must agree (the original bug), and match the expected rule.
    assert sem_m2 == run_m2 is expected
