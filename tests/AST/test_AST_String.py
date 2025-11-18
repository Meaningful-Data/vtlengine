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
    "complete_grammar.vtl",
    "GH_347.vtl",
]

params_prettier = [
    ("hierarchical_ruleset.vtl", "reference_hierarchical_ruleset.vtl"),
    ("group_all_time_aggr.vtl", "reference_group_all_time_aggr.vtl"),
    ("partition_by.vtl", "reference_partition_by.vtl"),
    ("join.vtl", "reference_join.vtl"),
    ("define_operator.vtl", "reference_define_operator.vtl"),
    ("define_dpr.vtl", "reference_define_dpr.vtl"),
    ("calc_case.vtl", "reference_calc_case.vtl"),
    ("case.vtl", "reference_case.vtl"),
    ("eval.vtl", "reference_eval.vtl"),
    ("calc_if_then_else.vtl", "reference_calc_if_then_else.vtl"),
    ("aggregate.vtl", "reference_aggregate.vtl"),
    ("assignment.vtl", "reference_assignment.vtl"),
    ("filter_cast_time_period.vtl", "reference_filter_cast_time_period.vtl"),
    ("calc_unary_op.vtl", "reference_calc_unary_op.vtl"),
    ("fill_time_series.vtl", "reference_fill_time_series.vtl"),
    ("date_add.vtl", "reference_date_add.vtl"),
    ("complete_grammar.vtl", "reference_complete_grammar.vtl"),
    ("HR_with_condition.vtl", "reference_HR_with_condition.vtl"),
    ("GH_352.vtl", "reference_GH_352.vtl")("GH_358.vtl", "reference_GH_358.vtl"),
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


@pytest.mark.parametrize("filename", params)
def test_syntax_validation_ast_string(filename):
    with open(vtl_filepath / filename, "r") as file:
        script = file.read()

    script_generated = ASTString().render(create_ast_with_comments(script))

    try:
        create_ast_with_comments(script)
    except Exception as e:
        pytest.fail(f"Syntax validation failed for original script {filename}: {e}")

    try:
        create_ast_with_comments(script_generated)
    except Exception as e:
        pytest.fail(f"Syntax validation failed for generated script {filename}: {e}")


@pytest.mark.parametrize("filename", params)
def test_syntax_validation_prettier(filename):
    with open(vtl_filepath / filename, "r") as file:
        script = file.read()

    script_generated = prettify(script)

    try:
        create_ast_with_comments(script)
    except Exception as e:
        pytest.fail(f"Syntax validation failed for original script {filename}: {e}")

    try:
        create_ast_with_comments(script_generated)
    except Exception as e:
        pytest.fail(f"Syntax validation failed for generated script {filename}: {e}")
