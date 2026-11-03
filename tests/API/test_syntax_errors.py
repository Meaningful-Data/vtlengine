import pandas as pd
import pytest

from vtlengine import run
from vtlengine.API import create_ast
from vtlengine.AST import VarID
from vtlengine.Exceptions import VTLSyntaxError

_EMPTY_DS = {"datasets": []}
_NO_DATA: dict[str, pd.DataFrame] = {}


def test_single_char_offending_token_has_caret_at_column():
    """`=` used in place of `:=` in a calc clause — caret points at the `=`."""
    script = "DS_A <- DS_1[calc test_2 = 42];"
    with pytest.raises(VTLSyntaxError) as ex:
        run(script=script, data_structures=_EMPTY_DS, datapoints=_NO_DATA)

    msg = str(ex.value)
    assert "line 1, column 26" in msg
    assert "DS_A <- DS_1[calc test_2 = 42];" in msg
    # 4-space indent + 25 spaces before column 26 → caret at offset 4 + 25 = 29
    assert "\n" + " " * 29 + "^" in msg
    # Single-character offending token = single caret
    assert "^^" not in msg


def test_multi_char_offending_token_is_fully_underlined():
    """A multi-character offending identifier gets a `^^^...` underline matching its length."""
    script = "DS_A := DS_1[calc x foo_bar 5];"
    with pytest.raises(VTLSyntaxError) as ex:
        run(script=script, data_structures=_EMPTY_DS, datapoints=_NO_DATA)

    msg = str(ex.value)
    assert "foo_bar" in msg
    # 7-character token → 7 carets
    assert "^^^^^^^" in msg


def test_error_on_line_two_aligns_to_that_line():
    """Multi-line script — preview shows the offending line, not line 1."""
    script = "DS_A := DS_1 + 1;\nDS_B := DS_2[calc x = 5];"
    with pytest.raises(VTLSyntaxError) as ex:
        run(script=script, data_structures=_EMPTY_DS, datapoints=_NO_DATA)

    msg = str(ex.value)
    assert "line 2" in msg
    assert "DS_B := DS_2[calc x = 5];" in msg
    # The first line of the script must NOT be quoted as the preview
    assert "    DS_A := DS_1 + 1;" not in msg


def test_tab_indented_script_keeps_caret_aligned():
    """Tabs in the source line are expanded so the caret aligns to the offending token."""
    script = "\t\tDS_A := DS_1[calc x = 1];"
    with pytest.raises(VTLSyntaxError) as ex:
        run(script=script, data_structures=_EMPTY_DS, datapoints=_NO_DATA)

    msg = str(ex.value)
    # Tabs should be expanded — the rendered line must not contain a raw \t
    assert "\t" not in msg
    # Locate the caret line; it must be a stretch of spaces followed by at least one ^.
    caret_line = msg.splitlines()[-1]
    assert caret_line.lstrip(" ") == "^"
    # And the character directly above the caret must be the `=`.
    preview_line = msg.splitlines()[-2]
    caret_col = len(caret_line) - len(caret_line.lstrip(" "))
    assert preview_line[caret_col] == "="


def test_valid_script_still_parses():
    """Regression guard — a well-formed script does not raise VTLSyntaxError."""
    data_structures = {
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
    datapoints = {"DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10.0, 20.0, 30.0]})}

    result = run(
        script="DS_A <- DS_1 * 2;",
        data_structures=data_structures,
        datapoints=datapoints,
    )
    assert "DS_A" in result


# ---------------------------------------------------------------------------
# IDENTIFIER digit/underscore handling (VTL 2.2 grammar)
# A bare identifier begins with a letter or `_` (e.g. `abc`, `_foo`, `_1abc`).
# A name that begins with a digit (e.g. `24A0`) must be quoted, except for a
# code item of a hierarchical rule (ITEM_CODE token). Purely numeric tokens
# (e.g. `123`) are constants, not identifiers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["24A0", "1abc", "9_foo"],
)
def test_bare_identifier_starting_with_digit_is_rejected(name):
    """Outside a hierarchical rule, a bare name cannot start with a digit."""
    script = f"DS_A <- {name} * 2;"
    with pytest.raises(VTLSyntaxError):
        run(script=script, data_structures=_EMPTY_DS, datapoints=_NO_DATA)


@pytest.mark.parametrize(
    "name",
    ["123", "0", "42"],
)
def test_purely_numeric_name_is_rejected(name):
    """A purely numeric token is a constant, not an identifier — it cannot be a target."""
    script = f"{name} <- DS_1;"
    with pytest.raises(VTLSyntaxError):
        run(script=script, data_structures=_EMPTY_DS, datapoints=_NO_DATA)


@pytest.mark.parametrize(
    "quoted_name",
    ["24A0", "1abc", "9_foo"],
)
def test_quoted_identifier_starting_with_digit_runs(quoted_name):
    """`'<digit-prefixed>'` is a valid IDENTIFIER and can name a dataset end-to-end."""
    script = f"DS_A <- '{quoted_name}' * 2;"
    data_structures = {
        "datasets": [
            {
                "name": quoted_name,
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Integer", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    result = run(script=script, data_structures=data_structures, datapoints=_NO_DATA)
    assert "DS_A" in result


@pytest.mark.parametrize(
    "underscore_name",
    ["_foo", "_unknown", "_1abc"],
)
def test_bare_underscore_prefixed_identifier_parses(underscore_name):
    """A bare identifier can start with `_`."""
    operand = create_ast(f"DS_A <- {underscore_name} * 2;").children[0].right.left
    assert isinstance(operand, VarID)
    assert operand.value == underscore_name


# ---------------------------------------------------------------------------
# Hierarchical rules
# A code item may start with a digit (e.g. `10A`), and the relation of a rule
# is one of `=`, `>`, `<`, `>=` and `<=` (hrComparisonOperand). `<>` and a
# missing relation are syntax errors, raised when the ruleset is defined (#1156).
# ---------------------------------------------------------------------------

_HR_DS = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}


def _hr_datapoints(items: list, values: list) -> dict:
    """One group per Id_1 value; each group holds the given items with the given values."""
    rows = [
        (group, item, value)
        for group, group_values in enumerate(values, start=1)
        for item, value in zip(items, group_values)
    ]
    return {"DS_1": pd.DataFrame(rows, columns=["Id_1", "Id_2", "Me_1"])}


def _check_hierarchy_bool_vars(rule: str, datapoints: dict, use_duckdb: bool) -> list:
    script = f"""
        define hierarchical ruleset HR_1 (valuedomain rule VD_1) is
            {rule}
        end hierarchical ruleset;

        DS_r <- check_hierarchy(DS_1, HR_1 rule Id_2 all);
    """
    result = run(
        script=script, data_structures=_HR_DS, datapoints=datapoints, use_duckdb=use_duckdb
    )
    return result["DS_r"].data.sort_values("Id_1")["bool_var"].tolist()


@pytest.mark.parametrize("use_duckdb", [False, True], ids=["pandas", "duckdb"])
def test_bare_code_item_starting_with_digit_runs(use_duckdb):
    """A hierarchical rule takes bare code items that start with a digit."""
    datapoints = _hr_datapoints(["10A", "10B", "1_0"], [[7.0, 3.0, 4.0], [8.0, 3.0, 4.0]])
    bool_vars = _check_hierarchy_bool_vars("R1: 10A = 10B + 1_0", datapoints, use_duckdb)
    assert bool_vars == [True, False]


@pytest.mark.parametrize("use_duckdb", [False, True], ids=["pandas", "duckdb"])
@pytest.mark.parametrize(
    ("relation", "expected"),
    [
        # TOTAL against A + B = 7, with TOTAL = 7, 10 and 5 in groups 1, 2 and 3
        ("=", [True, False, False]),
        (">", [False, True, False]),
        ("<", [False, False, True]),
        (">=", [True, True, False]),
        ("<=", [True, False, True]),
    ],
)
def test_hierarchical_rule_relation_runs(relation, expected, use_duckdb):
    """Each relation a hierarchical rule allows compares the code item with its sum."""
    datapoints = _hr_datapoints(
        ["TOTAL", "A", "B"], [[7.0, 3.0, 4.0], [10.0, 3.0, 4.0], [5.0, 3.0, 4.0]]
    )
    bool_vars = _check_hierarchy_bool_vars(f"R1: TOTAL {relation} A + B", datapoints, use_duckdb)
    assert bool_vars == expected


@pytest.mark.parametrize(
    ("ruleset", "column"),
    [
        pytest.param(
            "define hierarchical ruleset HR_1 (valuedomain rule VD_1) is\n"
            "    R1: TOTAL <> A + B\n"
            "end hierarchical ruleset;",
            15,
            id="not_equal",
        ),
        pytest.param(
            "define hierarchical ruleset HR_1 (valuedomain condition VD_2 as C_1 rule VD_1) is\n"
            '    R1: when C_1 = "X" then TOTAL <> A + B\n'
            "end hierarchical ruleset;",
            35,
            id="not_equal_in_conditioned_rule",
        ),
        pytest.param(
            "define hierarchical ruleset HR_1 (valuedomain rule VD_1) is\n"
            "    R1: TOTAL A + B\n"
            "end hierarchical ruleset;",
            15,
            id="missing_relation",
        ),
    ],
)
def test_hierarchical_rule_relation_is_rejected(ruleset, column):
    """`<>` or no relation in a hierarchical rule fails when the ruleset is defined."""
    with pytest.raises(VTLSyntaxError, match=f"line 2, column {column}:"):
        run(script=ruleset, data_structures=_EMPTY_DS, datapoints=_NO_DATA)
