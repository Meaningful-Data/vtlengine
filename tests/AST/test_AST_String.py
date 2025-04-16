from pathlib import Path

import pytest

from vtlengine.API import create_ast, prettify
from vtlengine.AST import Comment
from vtlengine.AST.ASTComment import create_ast_with_comments
from vtlengine.AST.ASTString import ASTString

base_path = Path(__file__).parent
vtl_filepath = base_path / "data" / "vtl"
prettier_path = base_path / "data" / "prettier"

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

params_prettier = [
    ("hierarchical_ruleset.vtl", "reference_hierarchical_ruleset.txt"),
    ("group_all_time_aggr.vtl", "reference_group_all_time_aggr.txt"),
    ("partition_by.vtl", "reference_partition_by.txt"),
    ("join.vtl", "reference_join.txt"),
    ("define_operator.vtl", "reference_define_operator.txt"),
    ("define_dpr.vtl", "reference_define_dpr.txt"),
    ("calc_case.vtl", "reference_calc_case.txt"),
    ("case.vtl", "reference_case.txt"),
    ("eval.vtl", "reference_eval.txt"),
    ("calc_if_then_else.vtl", "reference_calc_if_then_else.txt"),
    ("aggregate.vtl", "reference_aggregate.txt"),
    ("assignment.vtl", "reference_assignment.txt"),
    ("filter_cast_time_period.vtl", "reference_filter_cast_time_period.txt"),
    ("calc_unary_op.vtl", "reference_calc_unary_op.txt"),
    ("fill_time_series.vtl", "reference_fill_time_series.txt"),
    ("date_add.vtl", "reference_date_add.txt"),
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


def test_check_ast_string_output():
    with open(vtl_filepath / "comments.vtl", "r") as file:
        script = file.read()

    ast = create_ast_with_comments(script)
    result_script = ASTString().render(ast)

    assert script == result_script


def normalize_lines_in_text(script: str) -> str:
    return "\n".join(line.rstrip() for line in script.splitlines())


@pytest.mark.parametrize("filename, expected", params_prettier)
def test_prettier(filename, expected):
    with open(prettier_path / filename, "r") as file:
        script = file.read()
    result_script = prettify(script)
    with open(prettier_path / expected, "r") as file:
        expected_script = file.read()

    assert normalize_lines_in_text(result_script) == normalize_lines_in_text(expected_script)
