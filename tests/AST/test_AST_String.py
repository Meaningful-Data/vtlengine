from dataclasses import fields
from pathlib import Path

import pytest

from vtlengine.API import create_ast, prettify
from vtlengine.AST import AST, Comment
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
    "GH_352.vtl",
    "GH_358.vtl",
    "comments_end_line.vtl",
    "time_agg.vtl",
    "time_agg_ref.vtl",
    "time_agg_optional.vtl",
    "lag_lead_offset.vtl",
    "windowing_scalar_bound.vtl",
    "viral_propagation.vtl",
    "round_with_slash.vtl",
    "trunc_with_slash.vtl",
    "component_roles.vtl",
    "hierarchical_unquoted_codeitems.vtl",
    "join_body_clauses.vtl",
    "GH_987.vtl",
    "GH_1164.vtl",
    "GH_1179.vtl",
    "GH_1180.vtl",
    "GH_1181.vtl",
    "GH_1150.vtl",
    "GH_1166.vtl",
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
    ("GH_352.vtl", "reference_GH_352.vtl"),
    ("GH_358.vtl", "reference_GH_358.vtl"),
    ("validation.vtl", "reference_validation.vtl"),
    ("validation_str_errorlevel.vtl", "reference_validation_str_errorlevel.vtl"),
    ("unbounded.vtl", "reference_unbounded.vtl"),
    ("time_agg.vtl", "reference_time_agg.vtl"),
    ("time_agg_ref.vtl", "reference_time_agg_ref.vtl"),
    ("viral_propagation.vtl", "reference_viral_propagation.vtl"),
    ("GH_987.vtl", "reference_GH_987.vtl"),
    ("GH_1164.vtl", "reference_GH_1164.vtl"),
    ("GH_1179.vtl", "reference_GH_1179.vtl"),
    ("GH_1180.vtl", "reference_GH_1180.vtl"),
    ("GH_1181.vtl", "reference_GH_1181.vtl"),
    ("GH_1150.vtl", "reference_GH_1150.vtl"),
    ("GH_1166.vtl", "reference_GH_1166.vtl"),
]

POSITION_FIELDS = {"line_start", "column_start", "line_stop", "column_stop"}


def ast_difference(expected, actual, path="ast"):
    """Where two ASTs differ, or None when they are the same.

    AST.__eq__ skips children and Python takes 2 == 2.0, so neither tells whether a
    rendered script reads back as the script it came from.
    """
    if type(expected) is not type(actual):
        return f"{path}: {expected!r} != {actual!r}"
    if isinstance(expected, AST):
        for field in fields(expected):
            if field.name not in POSITION_FIELDS:
                difference = ast_difference(
                    getattr(expected, field.name),
                    getattr(actual, field.name),
                    f"{path}.{field.name}",
                )
                if difference is not None:
                    return difference
        return None
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return f"{path}: {len(expected)} != {len(actual)} elements"
        for i, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = ast_difference(expected_item, actual_item, f"{path}[{i}]")
            if difference is not None:
                return difference
        return None
    return None if expected == actual else f"{path}: {expected!r} != {actual!r}"


@pytest.mark.parametrize("filename", params)
def test_ast_string(filename):
    with open(vtl_filepath / filename, "r") as file:
        script = file.read()

    ast = create_ast(script)
    result_script = ASTString().render(ast)
    ast_result = create_ast(result_script)
    ast.children = sorted(ast.children, key=lambda x: x.__class__.__name__)
    ast_result.children = sorted(ast_result.children, key=lambda x: x.__class__.__name__)
    assert ast_difference(ast, ast_result) is None


@pytest.mark.parametrize("filename", params)
def test_ast_string_with_comments(filename):
    with open(vtl_filepath / filename, "r") as file:
        script = file.read()

    ast = create_ast_with_comments(script)
    result_script = ASTString().render(ast)
    ast_result = create_ast_with_comments(result_script)
    assert ast_difference(ast, ast_result) is None


partition_clauses = [
    "partition by id",
    "partition except anyo",
    "partition except anyo, id",
    "partition except all",
]


@pytest.mark.parametrize("clause", partition_clauses)
def test_ast_string_partition_clause(clause):
    """ASTString must preserve the partition operator (by / except / except all).

    Regression: ``partition except X`` used to be rendered as ``partition by X``,
    silently changing the analytic window when round-tripping script -> AST -> script.
    """
    script = f"DS_r <- DS_1[calc Me_2 := rank(over({clause} order by me))];"

    rendered = ASTString().render(create_ast(script))

    assert clause in rendered
    # Stable across a second round-trip
    assert clause in ASTString().render(create_ast(rendered))


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


def test_comments_end_line_parsing():
    with open(vtl_filepath / "comments_end_line.vtl", "r") as file:
        script = file.read()

    ast = create_ast_with_comments(script)
    assert len(ast.children) == 2
    assert isinstance(ast.children[1], Comment)
    assert ast.children[1].line_start == 2
    assert ast.children[1].line_stop == 2


@pytest.mark.parametrize("filename", ["comments.vtl", "GH_1181.vtl"])
def test_check_ast_string_output(filename):
    with open(vtl_filepath / filename, "r") as file:
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
    input_ast = create_ast(script)
    expected_input_ast = create_ast(expected_script)
    assert normalize_lines_in_text(result_script) == normalize_lines_in_text(expected_script)
    assert ast_difference(input_ast, expected_input_ast) is None


@pytest.mark.parametrize("filename, expected", params_prettier)
def test_prettier_no_trailing_whitespace(filename, expected):
    with open(prettier_path / filename, "r") as file:
        script = file.read()

    lines = prettify(script).splitlines()

    assert [line for line in lines if line != line.rstrip()] == []


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
