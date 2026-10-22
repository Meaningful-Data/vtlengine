"""
The class of an engine error and the family of its code agree (issue #1192): a SemanticError
carries a 1-* code and a RunTimeError a 2-* code.
"""

import ast
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from vtlengine.DataTypes import Integer
from vtlengine.Exceptions import DataLoadError, InputValidationException, check_key
from vtlengine.Exceptions.messages import centralised_messages
from vtlengine.Model import Component, Dataset, Role

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


def _plain_raises() -> List[str]:
    """Every raise of a plain Python error that a VTL error should replace (issue #1192)."""
    failures = []
    for path in sorted(SOURCE.rglob("*.py")):
        if "_cpp_parser" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            name = getattr(exc, "id", None) or getattr(exc, "attr", None)
            if name in ("Exception", "ValueError", "NotImplementedError"):
                failures.append(f"{path.relative_to(SOURCE)}:{node.lineno} {name}")
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


def test_load_errors_are_input_errors():
    error = DataLoadError("0-3-1-1", file="DS_1.csv")
    assert isinstance(error, InputValidationException)
    assert error.args[1] == "0-3-1-1"
    assert error.args[0].endswith(" Please check loaded file")
