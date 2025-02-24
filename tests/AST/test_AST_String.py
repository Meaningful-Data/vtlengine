from pathlib import Path

import pytest

from vtlengine.API import create_ast
from vtlengine.AST.ASTString import ASTString

base_path = Path(__file__).parent
vtl_filepath = base_path / "data" / "vtl"

params = [
    "analytic.vtl",
    "join_set.vtl",
    "library_items.vtl",
    "numbers.vtl",
    "other.vtl",
    "string.vtl",
    "time.vtl",
]


@pytest.mark.parametrize("filename", params)
def test_ast_string(filename):
    with open(vtl_filepath / filename, "r") as file:
        script = file.read()

    ast = create_ast(script)
    result_script = ASTString().render(ast)
    ast_result = create_ast(result_script)
    ast.children = sorted(ast.children, key=lambda x: x.__class__.__name__)
    ast_result.children = sorted(ast_result.children, key=lambda x: x.__class__.__name__)
    assert ast == ast_result
