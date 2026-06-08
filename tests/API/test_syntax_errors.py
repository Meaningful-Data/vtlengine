import pandas as pd
import pytest

from vtlengine import run
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
# IDENTIFIER digit/underscore handling
# The grammar accepts bare identifiers that:
#   - start with a digit as long as they contain at least one letter
#     (e.g. `24A0`, `1abc`, `9_foo`) — aligned with the VTL "regular names"
#     definition (VTL 2.1 User Manual, "The regular names");
#   - start with `_` (e.g. `_foo`).
# Purely numeric tokens (e.g. `123`) are NOT identifiers and cannot name an
# artefact; the same names are also valid when single-quoted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["24A0", "1abc", "9_foo"],
)
def test_bare_identifier_starting_with_digit_runs(name):
    """A bare digit-prefixed identifier (with a letter) can name a dataset end-to-end."""
    script = f"DS_A <- {name} * 2;"
    data_structures = {
        "datasets": [
            {
                "name": name,
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
def test_underscore_prefixed_identifier_runs(underscore_name):
    """A dataset named with a `_`-prefixed identifier resolves end-to-end."""
    script = f"DS_A <- {underscore_name} * 2;"
    data_structures = {
        "datasets": [
            {
                "name": underscore_name,
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Integer", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    result = run(script=script, data_structures=data_structures, datapoints=_NO_DATA)
    assert "DS_A" in result
