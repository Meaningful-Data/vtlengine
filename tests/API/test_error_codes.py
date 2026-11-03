"""
The class of an engine error and the family of its code agree (issue #1192): a SemanticError
carries a 1-* code and a RunTimeError a 2-* code.
"""

import ast
from pathlib import Path
from typing import Any, List
from unittest import mock

import pandas as pd
import pytest

from vtlengine import semantic_analysis
from vtlengine.AST.ASTConstructorModules.Terminals import Terminals
from vtlengine.DataTypes import Duration, Integer, ScalarType
from vtlengine.Exceptions import (
    DataLoadError,
    InputValidationException,
    RunTimeError,
    SemanticError,
    check_key,
    transformation_output,
)
from vtlengine.Exceptions.messages import centralised_messages
from vtlengine.Model import Component, DataComponent, Dataset, Role
from vtlengine.Operators import Operator
from vtlengine.Operators.Comparison import Between
from vtlengine.Operators.String import Parameterized as StringParameterized

SOURCE = Path(__file__).parents[2] / "src" / "vtlengine"
CODE_FAMILY = {"SemanticError": "1-", "RunTimeError": "2-"}


def _error_code_failures() -> List[str]:
    """Every SemanticError or RunTimeError built in the engine whose code does not fit it."""
    failures = []
    for path in sorted(SOURCE.rglob("*.py")):
        if "_cpp_parser" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            error_class = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if error_class not in CODE_FAMILY:
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            code_node = node.args[0] if node.args else keywords.get("code")
            site = f"{path.relative_to(SOURCE)}:{node.lineno} {error_class}"
            if not isinstance(code_node, ast.Constant) or not isinstance(code_node.value, str):
                # Only a literal code can be checked against its class
                failures.append(f"{site} {ast.unparse(code_node) if code_node else None}")
                continue
            code = code_node.value
            if not code.startswith(CODE_FAMILY[error_class]) or code not in centralised_messages:
                failures.append(f"{site} {code}")
    return failures


def _indirect_error_uses() -> List[str]:
    """Every use of SemanticError or RunTimeError other than building one, catching it or
    checking it: an error class held in a variable is built where its code cannot be read."""
    failures = []
    for path in sorted(SOURCE.rglob("*.py")):
        if "_cpp_parser" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        allowed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                allowed.add(id(node.func))
                if getattr(node.func, "id", None) in ("isinstance", "issubclass"):
                    allowed.update(id(arg) for arg in ast.walk(node.args[1]))
            elif isinstance(node, ast.ExceptHandler) and node.type is not None:
                allowed.update(id(sub) for sub in ast.walk(node.type))
            elif isinstance(node, ast.ClassDef):
                allowed.update(id(base) for base in node.bases)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
                allowed.update(id(sub) for sub in ast.walk(node.returns))
            elif isinstance(node, (ast.arg, ast.AnnAssign)) and node.annotation is not None:
                allowed.update(id(sub) for sub in ast.walk(node.annotation))
        for node in ast.walk(tree):
            name = getattr(node, "id", None) or getattr(node, "attr", None)
            if (
                name in CODE_FAMILY
                and isinstance(node, (ast.Name, ast.Attribute))
                and id(node) not in allowed
            ):
                failures.append(f"{path.relative_to(SOURCE)}:{node.lineno} {name}")
    return failures


def _hook_raises(tree: ast.AST) -> List[int]:
    """The raises that are the whole body of a method: hooks each subclass implements."""
    hooks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = [
                stmt
                for stmt in node.body
                if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
            ]
            if len(body) == 1 and isinstance(body[0], ast.Raise):
                hooks.append(id(body[0]))
    return hooks


def _plain_raises() -> List[str]:
    """Every raise of a plain Python error that a VTL error should replace (issue #1192).

    A NotImplementedError that is the whole body of a method declares a hook each subclass
    implements: reaching it is an engine bug, not a VTL error.
    """
    failures = []
    for path in sorted(SOURCE.rglob("*.py")):
        if "_cpp_parser" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hooks = _hook_raises(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            name = getattr(exc, "id", None) or getattr(exc, "attr", None)
            if name == "NotImplementedError" and id(node) in hooks:
                continue
            if name in ("Exception", "ValueError", "NotImplementedError", "SyntaxError"):
                failures.append(f"{path.relative_to(SOURCE)}:{node.lineno} {name}")
    return failures


def _input_error_code_failures() -> List[str]:
    """Every InputValidationException or DataLoadError built with a code outside 0-*."""
    failures = []
    for path in sorted(SOURCE.rglob("*.py")):
        if "_cpp_parser" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            error_class = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if error_class not in ("InputValidationException", "DataLoadError"):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            code_node = keywords.get("code")
            if code_node is None and error_class == "DataLoadError" and node.args:
                code_node = node.args[0]
            if code_node is None:
                continue
            site = f"{path.relative_to(SOURCE)}:{node.lineno} {error_class}"
            if not isinstance(code_node, ast.Constant) or not isinstance(code_node.value, str):
                failures.append(f"{site} {ast.unparse(code_node)}")
                continue
            code = code_node.value
            if not code.startswith("0-") or code not in centralised_messages:
                failures.append(f"{site} {code}")
    return failures


def test_error_codes_match_their_class():
    assert _error_code_failures() == []


def test_error_classes_are_built_directly():
    assert _indirect_error_uses() == []


def test_check_key_raises_input_error():
    with pytest.raises(InputValidationException) as context:
        check_key("type", ["Integer"], "Integr")
    assert context.value.args[1] == "0-1-1-13"
    assert "Did you mean Integer?" in context.value.args[0]


def test_empty_dataset_shape_mismatch_raises_input_error():
    components = {"Id_1": Component("Id_1", Integer, Role.IDENTIFIER, False)}
    left = Dataset("DS_1", components, pd.DataFrame(columns=["Id_1"]))
    right = Dataset("DS_2", components, pd.DataFrame(columns=["Id_1"]))
    # Data replaced after the construction, which checks it against the components
    right.data = pd.DataFrame(columns=["Id_1", "Me_1"])
    with pytest.raises(InputValidationException) as context:
        _ = left == right
    assert context.value.args[1] == "0-1-1-14"


def test_engine_raises_only_vtl_errors():
    assert _plain_raises() == []


@pytest.mark.parametrize(
    "hook",
    [
        lambda: Operator.validate(),
        lambda: Operator.evaluate(),
        lambda: ScalarType.implicit_cast("1", Integer),
        lambda: ScalarType.explicit_cast("1", Integer),
        lambda: StringParameterized.check_param(None, 1),
    ],
)
def test_hooks_each_subclass_implements_fail_fast(hook: Any) -> None:
    with pytest.raises(NotImplementedError):
        hook()


def test_load_errors_are_input_errors():
    error = DataLoadError("0-3-1-1", file="DS_1.csv")
    assert isinstance(error, InputValidationException)
    message = centralised_messages["0-3-1-1"]["message"].format(file="DS_1.csv")
    assert error.args == (f"{message} Please check loaded file", "0-3-1-1")
    with transformation_output("DS_r"):
        error = DataLoadError("0-3-1-1", comp_code="Me_1", file="DS_1.csv")
    assert error.args == (f"{message} Please check loaded file DS_r", "0-3-1-1")
    assert error.comp_code == "Me_1"


def test_input_error_codes_are_input_codes():
    assert _input_error_code_failures() == []


def test_duration_value_is_an_input_error():
    with pytest.raises(InputValidationException) as context:
        Duration.validate_duration("X")
    assert context.value.args[1] == "0-1-1-35"


DATA_STRUCTURES = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}


@pytest.mark.parametrize(
    "script, construct",
    [
        (
            "define operator f (x set<integer>) returns boolean is true end operator;",
            "set type 'set<integer>' (line 1, column 22)",
        ),
        (
            "define operator f (x hierarchical) returns boolean is true end operator;",
            "ruleset type 'hierarchical' (line 1, column 22)",
        ),
        (
            'DS_r <- eval(SQL1(DS_1) language "SQL" returns dataset '
            "{identifier<integer> Id_1, measure<number> _*});",
            "multiplicity modifier in 'measure<number> _*' (line 1, column 83)",
        ),
        (
            "DS_r <- DS_1[aggr Me_2 := sum(Me_1) group by Id_1 having true];",
            "having condition 'true' (line 1, column 58)",
        ),
        (
            "DS_r <- DS_1[aggr Me_2 := sum(Me_1) group by Id_1\n"
            "    having if sum(Me_1) > 0\n"
            "        then true else false];",
            "having condition 'if sum(Me_1) > 0 then true else false' (line 2, column 12)",
        ),
        (
            "DS_r <- DS_1[aggr Me_2 := sum(Me_1) group by Id_1 having if sum(Me_1) > 0 "
            "and sum(Me_1) < 100 and avg(Me_1) > 5 then true else false];",
            "having condition 'if sum(Me_1) > 0 and sum(Me_1) < 100 and avg(Me_1) > 5 th...' "
            "(line 1, column 58)",
        ),
    ],
)
def test_unsupported_constructs_quote_the_script(script: str, construct: str) -> None:
    with pytest.raises(SemanticError) as context:
        semantic_analysis(script=script, data_structures=DATA_STRUCTURES)
    assert context.value.args == (f"Unsupported VTL construct: {construct}.", "1-3-13")


@pytest.mark.parametrize(
    "script, construct",
    [
        (
            'DS_r <- DS_1[calc Me_2 := cast(Me_1, string, "YYYY")];',
            'cast with the mask "YYYY"',
        ),
        ("DS_r <- DS_1[pivot Id_1, Me_1];", "pivot clause"),
    ],
)
def test_unsupported_operations_name_the_transformation(script: str, construct: str) -> None:
    with pytest.raises(SemanticError) as context:
        semantic_analysis(script=script, data_structures=DATA_STRUCTURES)
    assert context.value.args == (
        f"Unsupported VTL construct: {construct}. Please check transformation with output "
        "Dataset DS_r",
        "1-3-13",
    )


def test_invalid_parameter_type_quotes_the_script():
    script = "define operator f (x my_vd) returns integer is 1 end operator;"
    with pytest.raises(InputValidationException) as context:
        semantic_analysis(script=script, data_structures=DATA_STRUCTURES)
    assert context.value.args == (
        "Invalid parameter type definition 'my_vd' (line 1, column 22).",
        "0-5-1-10",
    )


@pytest.mark.parametrize(
    "window, message",
    [
        (
            "data points between unbounded preceding and unbounded preceding",
            "Cannot have 2 preceding clauses with unbounded in analytic clause "
            "'data points between unbounded preceding and unbounded pre...' (line 1, column 56).",
        ),
        (
            "range between unbounded preceding and unbounded preceding",
            "Cannot have 2 preceding clauses with unbounded in analytic clause "
            "'range between unbounded preceding and unbounded preceding' (line 1, column 56).",
        ),
        (
            "data points between unbounded following and unbounded following",
            "Cannot have 2 following clauses with unbounded in analytic clause "
            "'data points between unbounded following and unbounded fol...' (line 1, column 56).",
        ),
        (
            "range between unbounded following and unbounded following",
            "Cannot have 2 following clauses with unbounded in analytic clause "
            "'range between unbounded following and unbounded following' (line 1, column 56).",
        ),
    ],
)
def test_unbounded_windows_quote_the_script(window: str, message: str) -> None:
    script = f"DS_r <- DS_1[calc Me_2 := sum(Me_1 over (order by Id_1 {window}))];"
    with pytest.raises(InputValidationException) as context:
        semantic_analysis(script=script, data_structures=DATA_STRUCTURES)
    assert context.value.args[0] == message


def test_error_code_errors_quote_the_script(monkeypatch: pytest.MonkeyPatch) -> None:
    script = (
        'define datapoint ruleset dpr (variable Me_1) is r1: isnull(Me_1) errorcode "E1" '
        "end datapoint ruleset;"
    )
    unreadable = UnboundLocalError("constant_node")
    monkeypatch.setattr(Terminals, "visitConstant", mock.Mock(side_effect=unreadable))
    with pytest.raises(InputValidationException) as context:
        semantic_analysis(script=script, data_structures=DATA_STRUCTURES)
    assert context.value.args == (
        "Error code must be a string: '\"E1\"' (line 1, column 76).",
        "0-5-1-5",
    )


def test_between_bounds_of_different_lengths_is_a_runtime_error():
    operand = DataComponent("Me_1", pd.Series([1, 2]), Integer, Role.MEASURE, True)
    from_ = DataComponent("Me_2", pd.Series([0, 0]), Integer, Role.MEASURE, True)
    to = DataComponent("Me_3", pd.Series([5]), Integer, Role.MEASURE, True)
    with pytest.raises(RunTimeError) as context:
        Between.evaluate(operand, from_, to)
    assert context.value.args[1] == "2-1-1-18"
