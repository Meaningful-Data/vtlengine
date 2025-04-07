from pathlib import Path

import pytest

from vtlengine.API import create_ast
from vtlengine.AST import Comment
from vtlengine.AST.ASTComment import create_ast_with_comments
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
    "comments.vtl",
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


@pytest.mark.parametrize("filename", params)
def test_ast_string_with_comments(filename):
    with open(vtl_filepath / filename, "r") as file:
        script = file.read()

    ast = create_ast_with_comments(script)
    result_script = ASTString().render(ast)
    ast_result = create_ast_with_comments(result_script)
    assert ast == ast_result


def test_comments_parsing():
    with open(vtl_filepath / "comments.vtl", "r") as file:
        script = file.read()

    ast = create_ast_with_comments(script)
    for i in [0, 2, 4, 6, 8]:
        assert isinstance(ast.children[i], Comment)
    for i in [1, 3, 5, 7]:
        assert not isinstance(ast.children[i], Comment)

    assert ast.children[0].value == "// Line comment before first transformation"
    assert "/*" in ast.children[2].value[:2]
    assert "*/" in ast.children[2].value[-2:]
    assert ast.children[2].line_start == 3
    assert ast.children[2].line_stop == 6
