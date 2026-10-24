# ruff: noqa: S101
"""Checks for the import of the VTL 2.2 manual's examples.

The checks on the real manual need a clone of ``sdmx-twg/vtl`` at the pinned commit. They read
its path from the ``VTL_MANUAL_SPEC`` environment variable and are skipped when it is unset or
the folder does not exist.

To get the clone (a sparse checkout of ``v2.2/docs`` at the pinned commit)::

    git clone -q --filter=blob:none --sparse https://github.com/sdmx-twg/vtl.git <dir>
    git -C <dir> checkout -q 140e946ed3f888951adb8bb66ea45030be84e2aa
    git -C <dir> sparse-checkout set v2.2/docs

and pass the manual's folder::

    VTL_MANUAL_SPEC=<dir>/v2.2/docs/reference_manual
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import pytest

from tool.examples import ExampleError, ExpectedScalar, load_example
from tool.importer import (
    ManualImportError,
    build_index_source,
    convert_structure,
    import_chapter,
    import_example,
)
from tool.model import DL_PAGES, Chapter, Operator, Variant, authored, manual

COMMIT = "140e946ed3f888951adb8bb66ea45030be84e2aa"
PAGE = "Fake chapter/Fake page"


@pytest.fixture
def manual_spec() -> Path:
    """The manual's folder, from ``VTL_MANUAL_SPEC``; the check is skipped without it."""
    folder = os.environ.get("VTL_MANUAL_SPEC", "")
    if not folder or not Path(folder).is_dir():
        pytest.skip("set VTL_MANUAL_SPEC to a clone of sdmx-twg/vtl v2.2/docs/reference_manual")
    return Path(folder)


# What the importer reports on the manual at the pinned commit: the examples that need a hand,
# each with a word of every warning. No other example of the 191 has a warning.
WARNED_EXAMPLES = {
    # The expected dataset is named DS_3, the script assigns DS_r.
    "Aggregate and Analytic operators/Aggregate invocation/ex_4": ["DS_3"],
    # ds_2.json declares DS_1, a slip: the scripts and the text of the pages say DS_2.
    "Clause operators/Unpivoting/ex_2": ["ds_2.json"],
    "General purpose operators/Membership/ex_4": ["ds_2.json"],
    "General purpose operators/Membership/ex_5": ["ds_2.json"],
    "General purpose operators/Membership/ex_6": ["ds_2.json"],
    "General purpose operators/Membership/ex_8": ["ds_2.json"],
    # ds_5.json and ds_6.json declare DS_r, the result: the script reads DS_5 and DS_6.
    "Join operators/Inner Join/ex_5": ["ds_5.json", "ds_6.json"],
    # The ruleset and the value domain are defined by the text of the page only.
    "Comparison operators/Element of/ex_3": ["myGeoValueDomain"],
    "Hierarchical aggregation/Hierarchical roll-up/ex_1": ["HR_1"],
    "Hierarchical aggregation/Hierarchical roll-up/ex_2": ["HR_1"],
    "Hierarchical aggregation/Hierarchical roll-up/ex_3": ["HR_1"],
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dataset(name: str, *components: tuple[str, str, str]) -> str:
    """The manual's structure file of a dataset, from ``(name, role, data_type)`` components."""
    items = [{"name": n, "role": r, "data_type": t} for n, r, t in components]
    return json.dumps({"name": name, "components": items})


ID_ME = (("Id_1", "Identifier", "Integer"), ("Me_1", "Measure", "Integer"))

# A page whose only example is ``DS_r := DS_1 + 1``.
ADD_ONE = {
    "ds_1.json": dataset("DS_1", *ID_ME),
    "ds_1.csv": "Id_1,Me_1\n1,10\n2,20\n",
    "ex_1.vtl": "DS_r := DS_1 + 1;",
    "ex_1.json": dataset("DS_r", *ID_ME),
    "ex_1.csv": "Id_1,Me_1\n1,11\n2,21\n",
}


def fake_manual(root: Path, files: dict[str, str], page: str = PAGE) -> Path:
    """A manual folder whose ``page`` has ``files`` as its examples; returns the folder."""
    write(root / "operators" / page / "index.rst", "page\n")
    for name, text in files.items():
        write(root / "operators" / page / "examples" / name, text)
    return root


def import_script(tmp_path: Path, script: str, **extra: str) -> list[str]:
    """Import ``ADD_ONE`` with another script (and more files); return the warnings."""
    files = {**ADD_ONE, "ex_1.vtl": script, **extra}
    spec = fake_manual(tmp_path / "spec", files)
    return import_example(spec, PAGE, "ex_1", tmp_path / "out")


def parse_index(source: str) -> dict[str, Any]:
    """The constants a generated ``manual_index.py`` assigns, read without running it."""
    values: dict[str, Any] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assert node.value is not None
            values[node.target.id] = ast.literal_eval(node.value)
    return values


# --- Conversion of the structures ---


def test_convert_structure_time_period_and_nullable() -> None:
    manual_json: dict[str, Any] = {
        "name": "DS_1",
        "components": [
            {"name": "Id_1", "role": "Identifier", "data_type": "TimePeriod"},
            {"name": "Me_1", "role": "Measure", "data_type": "Number"},
            {"name": "At_1", "role": "Attribute", "data_type": "String"},
            {"name": "At_2", "role": "ViralAttribute", "data_type": "boolean"},
        ],
    }
    converted = convert_structure(manual_json)
    assert converted == {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Time_Period",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                    {"name": "At_1", "type": "String", "role": "Attribute", "nullable": True},
                    {"name": "At_2", "type": "Boolean", "role": "ViralAttribute", "nullable": True},
                ],
            }
        ]
    }
    first = converted["datasets"][0]["DataStructure"][0]
    assert list(first) == ["name", "type", "role", "nullable"]
    assert manual_json["components"][0]["data_type"] == "TimePeriod"  # the argument is untouched


@pytest.mark.parametrize(
    ("manual_type", "expected"),
    [
        ("Integer", "Integer"),
        ("Number", "Number"),
        ("String", "String"),
        ("Boolean", "Boolean"),
        ("Date", "Date"),
        ("Time", "Time"),
        ("TimePeriod", "Time_Period"),
        ("Duration", "Duration"),
        ("integer", "Integer"),
        ("string", "String"),
    ],
)
def test_convert_structure_canonical_types(manual_type: str, expected: str) -> None:
    manual_json = {
        "name": "DS_1",
        "components": [{"name": "Me_1", "role": "Measure", "data_type": manual_type}],
    }
    assert convert_structure(manual_json)["datasets"][0]["DataStructure"][0]["type"] == expected


@pytest.mark.parametrize(
    "manual_json",
    [
        {"name": "DS_1", "components": [{"name": "Me_1", "role": "Measure", "data_type": "Text"}]},
        {"name": "DS_1", "components": [{"name": "Me_1", "role": "Key", "data_type": "String"}]},
        {"name": "DS_1", "components": [{"name": "Me_1", "role": "Measure"}]},
        {"name": "DS_1"},
        {"components": []},
    ],
)
def test_convert_structure_refuses_what_it_cannot_read(manual_json: dict[str, Any]) -> None:
    with pytest.raises(ManualImportError):
        convert_structure(manual_json)


def test_convert_scalar_expected(tmp_path: Path) -> None:
    assert convert_structure({"name": "SC_r", "type": "integer"}) == {
        "scalars": [{"name": "SC_r", "type": "Integer"}]
    }

    files = {
        "ds_1.json": dataset("DS_1", *ID_ME),
        "ds_1.csv": "Id_1,Me_1\n1,10\n2,3\n",
        "ex_1.vtl": "SC_r := max(DS_1)#Me_1;",
        "ex_1.json": json.dumps({"name": "SC_r", "type": "integer"}),
        "ex_1.csv": "SC_r\n 10  \n",
    }
    dest = tmp_path / "out"
    assert import_example(fake_manual(tmp_path / "spec", files), PAGE, "ex_1", dest) == []

    assert json.loads((dest / "expected/scalars.json").read_text(encoding="utf-8")) == {
        "scalars": [{"name": "SC_r", "type": "Integer", "value": "10"}]
    }
    assert sorted(p.name for p in (dest / "expected").iterdir()) == ["scalars.json"]
    assert load_example(dest).expected_scalars == [ExpectedScalar("SC_r", "Integer", "10")]


SCALAR_FILES = {
    **ADD_ONE,
    "ex_1.vtl": "SC_r := DS_1#Me_1;",
    "ex_1.json": json.dumps({"name": "SC_r", "type": "string"}),
}


def scalar_value(tmp_path: Path, csv_text: str) -> Any:
    """The ``value`` that the import writes for a scalar result whose data file is ``csv_text``."""
    spec = fake_manual(tmp_path / "spec", {**SCALAR_FILES, "ex_1.csv": csv_text})
    dest = tmp_path / "out"
    assert import_example(spec, PAGE, "ex_1", dest) == []
    document = json.loads((dest / "expected/scalars.json").read_text(encoding="utf-8"))
    return document["scalars"][0]["value"]


def test_scalar_string_value_keeps_its_spaces(tmp_path: Path) -> None:
    assert scalar_value(tmp_path, 'SC_r\n"  P "\n') == "  P "


def test_scalar_without_a_value_is_null(tmp_path: Path) -> None:
    assert scalar_value(tmp_path, "SC_r\n\n") is None
    assert load_example(tmp_path / "out").expected_scalars == [
        ExpectedScalar("SC_r", "String", None)
    ]


def test_scalar_needs_a_data_row(tmp_path: Path) -> None:
    spec = fake_manual(tmp_path / "spec", {**SCALAR_FILES, "ex_1.csv": "SC_r\n"})
    with pytest.raises(ManualImportError, match="value"):
        import_example(spec, PAGE, "ex_1", tmp_path / "out")


# --- Import of one example ---


def test_import_writes_the_example_folder(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    assert import_example(fake_manual(tmp_path / "spec", ADD_ONE), PAGE, "ex_1", dest) == []

    assert (dest / "script.vtl").read_text(encoding="utf-8") == "DS_r := DS_1 + 1;\n"
    assert sorted(p.name for p in (dest / "input").iterdir()) == ["DS_1.csv", "DS_1.json"]
    assert sorted(p.name for p in (dest / "expected").iterdir()) == ["DS_r.csv", "DS_r.json"]
    assert (dest / "input/DS_1.csv").read_text(encoding="utf-8") == "Id_1,Me_1\n1,10\n2,20\n"
    assert json.loads((dest / "input/DS_1.json").read_text(encoding="utf-8")) == {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Integer", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    # The manual does not say whether a result is nullable: the expected structure leaves it open.
    expected = load_example(dest).expected_datasets[0]
    assert expected.name == "DS_r"
    assert {name: c.nullable for name, c in expected.components.items()} == {
        "Id_1": None,
        "Me_1": None,
    }
    assert expected.rows == [{"Id_1": "1", "Me_1": "11"}, {"Id_1": "2", "Me_1": "21"}]


def test_import_picks_only_referenced_unassigned_inputs(tmp_path: Path) -> None:
    other = {
        "ds_2.json": dataset("DS_r", *ID_ME),
        "ds_2.csv": "Id_1,Me_1\n1,11\n",
        "ds_3.json": dataset("DS_3", *ID_ME),
        "ds_3.csv": "Id_1,Me_1\n1,30\n",
    }
    dest = tmp_path / "out"
    spec = fake_manual(tmp_path / "spec", {**ADD_ONE, **other})
    assert import_example(spec, PAGE, "ex_1", dest) == []

    assert sorted(p.name for p in (dest / "input").iterdir()) == ["DS_1.csv", "DS_1.json"]
    assert [file.name for file in load_example(dest).structures] == ["DS_1.json"]


def test_import_reads_a_capitalised_input_file(tmp_path: Path) -> None:
    files = {**ADD_ONE, "DS_1.json": ADD_ONE["ds_1.json"], "DS_1.csv": ADD_ONE["ds_1.csv"]}
    del files["ds_1.json"], files["ds_1.csv"]
    dest = tmp_path / "out"
    assert import_example(fake_manual(tmp_path / "spec", files), PAGE, "ex_1", dest) == []
    assert sorted(p.name for p in (dest / "input").iterdir()) == ["DS_1.csv", "DS_1.json"]


def test_import_strips_cells_of_non_string_columns_only(tmp_path: Path) -> None:
    columns = (
        ("Id_1", "Identifier", "Integer"),
        ("Me_1", "Measure", "Number"),
        ("Me_2", "Measure", "String"),
    )
    bom = "\ufeff"
    rows = [
        f'{bom}Id_1,Me_1,Me_2\r\n1 ,2.1     ," hi, there "\r\n',
        '2,  3,"hello "\r\n',
        '3,,"say ""hi"""\r\n',
        "4,5,plain\r\n",
    ]
    files = {
        "ds_1.json": dataset("DS_1", *columns),
        "ds_1.csv": "".join(rows),
        "ex_1.vtl": "DS_r := DS_1;",
        "ex_1.json": dataset("DS_r", *columns),
        "ex_1.csv": "Id_1,Me_1,Me_2\n1,2.1,x \n",
    }
    dest = tmp_path / "out"
    assert import_example(fake_manual(tmp_path / "spec", files), PAGE, "ex_1", dest) == []

    # No BOM, one LF per line; String cells keep their spaces (quoted so that they show).
    assert (dest / "input/DS_1.csv").read_bytes() == (
        b'Id_1,Me_1,Me_2\n1,2.1," hi, there "\n2,3,"hello "\n3,,"say ""hi"""\n4,5,plain\n'
    )
    assert (dest / "expected/DS_r.csv").read_bytes() == b'Id_1,Me_1,Me_2\n1,2.1,"x "\n'
    example = load_example(dest)
    assert example.expected_datasets[0].rows == [{"Id_1": "1", "Me_1": "2.1", "Me_2": "x "}]


def test_import_refuses_data_that_does_not_match_its_structure(tmp_path: Path) -> None:
    files = {**ADD_ONE, "ds_1.csv": "Id_1,Me_9\n1,10\n"}
    with pytest.raises(ManualImportError, match="Me_9"):
        import_example(fake_manual(tmp_path / "spec", files), PAGE, "ex_1", tmp_path / "out")
    files = {**ADD_ONE, "ex_1.csv": "Id_1,Me_1\n1\n"}
    with pytest.raises(ManualImportError, match="fields"):
        import_example(fake_manual(tmp_path / "other", files), PAGE, "ex_1", tmp_path / "out")
    assert not (tmp_path / "out").exists()  # nothing is written when the example cannot be read


@pytest.mark.parametrize("name", ["../DS_r", "DS_r/x", "DS-r", "DS.r", "DS r"])
def test_import_refuses_a_result_name_that_cannot_name_a_file(tmp_path: Path, name: str) -> None:
    # The name of the result names the files of ``expected/``: it must stay a plain name.
    files = {**ADD_ONE, "ex_1.json": dataset(name, *ID_ME)}
    spec = fake_manual(tmp_path / "spec", files)
    with pytest.raises(ManualImportError, match="cannot name a file"):
        import_example(spec, PAGE, "ex_1", tmp_path / "out" / "example")
    assert not (tmp_path / "out").exists()  # nothing is written, inside the folder or next to it


def test_import_names_an_input_after_its_file_when_it_declares_another_name(tmp_path: Path) -> None:
    # The manual's ds_2.json declares DS_1, a slip: the script and the page text say DS_2.
    files = {
        **ADD_ONE,
        "ds_2.json": dataset("DS_1", *ID_ME),
        "ds_2.csv": "Id_1,Me_1\n1,5\n",
    }
    spec = fake_manual(tmp_path / "spec", files)

    one = tmp_path / "one"
    assert import_example(spec, PAGE, "ex_1", one) == []  # DS_1 is ds_1.json's, nothing is lost
    assert sorted(p.name for p in (one / "input").iterdir()) == ["DS_1.csv", "DS_1.json"]
    assert (one / "input/DS_1.csv").read_text(encoding="utf-8") == "Id_1,Me_1\n1,10\n2,20\n"

    write(spec / "operators" / PAGE / "examples/ex_1.vtl", "DS_r := DS_1 + DS_2;")
    two = tmp_path / "two"
    warnings = import_example(spec, PAGE, "ex_1", two)
    assert sorted(p.name for p in (two / "input").iterdir()) == [
        "DS_1.csv",
        "DS_1.json",
        "DS_2.csv",
        "DS_2.json",
    ]
    assert (two / "input/DS_2.csv").read_text(encoding="utf-8") == "Id_1,Me_1\n1,5\n"
    declared = json.loads((two / "input/DS_2.json").read_text(encoding="utf-8"))
    assert declared["datasets"][0]["name"] == "DS_2"
    assert len(warnings) == 1
    assert "ds_2.json" in warnings[0]
    assert "DS_1" in warnings[0]
    assert "DS_2" in warnings[0]


def test_import_refuses_two_files_for_one_dataset(tmp_path: Path) -> None:
    files = {**ADD_ONE, "DS_1.json": ADD_ONE["ds_1.json"], "DS_1.csv": ADD_ONE["ds_1.csv"]}
    with pytest.raises(ManualImportError, match="DS_1"):
        import_example(fake_manual(tmp_path / "spec", files), PAGE, "ex_1", tmp_path / "out")


# --- Warnings ---


def test_warns_on_referenced_dataset_without_input_file(tmp_path: Path) -> None:
    warnings = import_script(tmp_path, "DS_r := DS_1 + DS_9;")
    assert len(warnings) == 1
    assert "DS_9" in warnings[0]
    assert "DS_1" not in warnings[0]


def test_no_warning_for_a_dataset_the_script_assigns(tmp_path: Path) -> None:
    script = "DS_2 := DS_1 + 1;\nDS_r <- DS_2 * 2; DS_3 := DS_r;"
    assert import_script(tmp_path, script) == []


def test_warns_when_the_expected_dataset_is_not_assigned(tmp_path: Path) -> None:
    warnings = import_script(
        tmp_path, "DS_r := DS_1 + 1;", **{"ex_1.json": dataset("DS_3", *ID_ME)}
    )
    assert len(warnings) == 1
    assert "DS_3" in warnings[0]


@pytest.mark.parametrize(
    ("script", "undefined"),
    [
        ("DS_r := hierarchy(DS_1, HR_1 rule Id_2);", ["HR_1"]),
        ("DS_r := hierarchy ( DS_1, HR_1 rule Id_2 non_null );", ["HR_1"]),
        ("DS_r := check_hierarchy(DS_1, HR_1 rule Id_2 partial_null all);", ["HR_1"]),
        ("DS_r := check_datapoint(DS_1, dpr1 all);", ["dpr1"]),
        ('DS_r := hierarchy(DS_1[filter Id_2 in {"A", "B"}], HR_2 rule Id_2);', ["HR_2"]),
        ("DS_r := hierarchy(DS_1[calc Me_2 := nvl(Me_1, 0)], HR_3);", ["HR_3"]),
        (
            "define hierarchical ruleset HR_1 ( valuedomain rule VD_1 ) is\n"
            "    A = B + C\nend hierarchical ruleset;\n"
            "DS_r := hierarchy(DS_1, HR_1 rule Id_2);",
            [],
        ),
        (
            "DS_r := check_datapoint(DS_1, dpr1);define datapoint ruleset dpr1 ( variable Me_1 )"
            " is\n    Me_1 >= 0\nend datapoint ruleset;",
            [],
        ),
        ("DS_r := DS_1 + 1; /* hierarchy(DS_1, HR_9) */", []),
    ],
)
def test_warns_on_undefined_ruleset(tmp_path: Path, script: str, undefined: list[str]) -> None:
    warnings = import_script(tmp_path, script)
    assert len(warnings) == len(undefined)
    for name, warning in zip(undefined, warnings):
        assert name in warning
        assert "ruleset" in warning


@pytest.mark.parametrize(
    ("script", "undefined"),
    [
        ("DS_r := DS_1#Id_2 in myGeoValueDomain;", ["myGeoValueDomain"]),
        ("DS_r := DS_1 [ filter Id_2 not_in geo_vd ];", ["geo_vd"]),
        ("DS_r := DS_1 in { 0, 3, 6, 12 };", []),
        ("DS_r := DS_1 [ calc Me_2 := Me_1 in {0, 3} ];", []),
        ("DS_r := exists_in(DS_1, DS_1, all);", []),
        ('DS_r := DS_1 [ calc Me_2 := "ends in vd" ];', []),
    ],
)
def test_warns_on_value_domain_not_defined(
    tmp_path: Path, script: str, undefined: list[str]
) -> None:
    warnings = import_script(tmp_path, script)
    assert len(warnings) == len(undefined)
    for name, warning in zip(undefined, warnings):
        assert name in warning
        assert "value domain" in warning


# --- Existing destinations ---


def test_import_does_not_overwrite_unless_forced(tmp_path: Path) -> None:
    spec = fake_manual(tmp_path / "spec", ADD_ONE)
    dest = tmp_path / "out" / "example"
    assert import_example(spec, PAGE, "ex_1", dest) == []

    corrected = dest / "expected/DS_r.csv"
    write(corrected, "Id_1,Me_1\n1,99\n")
    write(dest / "notes.md", "an erratum\n")
    write(dest / "input/DS_9.csv", "stale\n")

    skipped = import_example(spec, PAGE, "ex_1", dest)
    assert len(skipped) == 1
    assert "exists" in skipped[0]
    assert corrected.read_text(encoding="utf-8") == "Id_1,Me_1\n1,99\n"
    assert (dest / "input/DS_9.csv").is_file()

    assert import_example(spec, PAGE, "ex_1", dest, force=True) == []
    assert corrected.read_text(encoding="utf-8") == "Id_1,Me_1\n1,11\n2,21\n"
    assert not (dest / "input/DS_9.csv").exists()  # the files of a forced import are its own
    assert (dest / "notes.md").read_text(encoding="utf-8") == "an erratum\n"  # the notes are kept


def test_import_into_a_folder_that_has_no_example_files(tmp_path: Path) -> None:
    spec = fake_manual(tmp_path / "spec", ADD_ONE)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert import_example(spec, PAGE, "ex_1", empty) == []
    assert (empty / "script.vtl").is_file()

    noted = tmp_path / "noted"
    write(noted / "notes.md", "written first\n")
    write(noted / "value_domains/vd.json", "{}")
    assert import_example(spec, PAGE, "ex_1", noted) == []
    assert (noted / "script.vtl").is_file()
    assert (noted / "notes.md").read_text(encoding="utf-8") == "written first\n"
    assert (noted / "value_domains/vd.json").is_file()


# --- A chapter ---


def test_import_chapter_imports_the_manual_examples_of_the_catalogue(tmp_path: Path) -> None:
    spec = fake_manual(tmp_path / "spec", ADD_ONE)
    variants = (
        Variant("ds", "dataset", (manual("ex_1"), authored())),
        Variant("sc", "scalar", (authored(),)),
    )
    plus = Operator("plus", "Plus", "+", PAGE, ("number",), ("number",), variants)
    chapter = Chapter("fake", "Fake chapter", (plus,))
    root = tmp_path / "examples"

    report = import_chapter(chapter, spec, root)
    assert (report.imported, report.skipped, report.warnings) == (1, [], [])
    assert load_example(root / "fake/plus/ds/manual_ex_1").script == "DS_r := DS_1 + 1;\n"
    assert not (root / "fake/plus/ds/minimal").exists()  # authored examples are not imported
    assert not (root / "fake/plus/sc").exists()

    again = import_chapter(chapter, spec, root)
    assert (again.imported, again.skipped, again.warnings) == (0, ["fake/plus/ds/manual_ex_1"], [])

    forced = import_chapter(chapter, spec, root, force=True)
    assert (forced.imported, forced.skipped, forced.warnings) == (1, [], [])


# --- Index of the manual ---


def test_build_index_source_lists_the_pages_and_examples(tmp_path: Path) -> None:
    for name in [
        "Chapter A/index.rst",  # the index of a chapter is not a page
        "Chapter A/Page 1/index.rst",
        "Chapter A/Page 1/examples/ex_1.vtl",
        "Chapter A/Page 1/examples/ex_10.vtl",
        "Chapter A/Page 1/examples/ex_2.vtl",
        "Chapter A/Page 1/examples/ex_1.json",  # the structure of an example, not an example
        "Chapter A/Page 1/examples/ds_1.vtl",
        "Chapter A/Page 2/index.rst",  # a page without examples
        "Chapter A/Notes/content.rst",  # a folder without an index is not a page
        "Chapter B/Page 3/index.rst",
        "Chapter B/Page 3/examples/ex_1.vtl",
    ]:
        write(tmp_path / "operators" / name, "")
    write(tmp_path / "operators/index.rst", "")

    source = build_index_source(tmp_path, COMMIT)
    assert parse_index(source) == {
        "MANUAL_COMMIT": COMMIT,
        "MANUAL_PAGES": tuple(
            sorted(["Chapter A/Page 1", "Chapter A/Page 2", "Chapter B/Page 3", *DL_PAGES])
        ),
        "MANUAL_EXAMPLES": (
            "Chapter A/Page 1/ex_1",
            "Chapter A/Page 1/ex_10",
            "Chapter A/Page 1/ex_2",
            "Chapter B/Page 3/ex_1",
        ),
    }
    assert build_index_source(tmp_path, COMMIT) == source  # nothing varies between runs
    assert source.endswith("\n")
    assert "\r" not in source
    assert "\ufeff" not in source


@pytest.mark.parametrize("commit", ["", "main", "140e946", COMMIT.upper(), COMMIT + "0"])
def test_build_index_source_needs_a_full_commit(tmp_path: Path, commit: str) -> None:
    write(tmp_path / "operators/Chapter A/Page 1/index.rst", "")
    with pytest.raises(ManualImportError, match="commit"):
        build_index_source(tmp_path, commit)


def test_build_index_source_needs_a_manual(tmp_path: Path) -> None:
    with pytest.raises(ManualImportError, match="operators"):
        build_index_source(tmp_path, COMMIT)


def test_index_counts(manual_spec: Path) -> None:
    index = parse_index(build_index_source(manual_spec, COMMIT))
    pages, examples = index["MANUAL_PAGES"], index["MANUAL_EXAMPLES"]
    assert index["MANUAL_COMMIT"] == COMMIT
    assert len(pages) == 104
    assert len(examples) == 191
    assert pages == tuple(sorted(set(pages)))
    assert examples == tuple(sorted(set(examples)))
    assert set(DL_PAGES) <= set(pages)
    assert {example.rsplit("/", 1)[0] for example in examples} <= set(pages)
    assert "Clause operators/Unpivoting/ex_2" in examples


def test_every_manual_example_imports_and_loads(manual_spec: Path, tmp_path: Path) -> None:
    index = parse_index(build_index_source(manual_spec, COMMIT))
    unreadable: list[str] = []
    warned: dict[str, list[str]] = {}
    for key in index["MANUAL_EXAMPLES"]:
        page, ref = key.rsplit("/", 1)
        dest = tmp_path / key
        warnings = import_example(manual_spec, page, ref, dest)
        if warnings:
            warned[key] = warnings
        try:
            load_example(dest)
        except ExampleError as exc:
            unreadable.append(str(exc))
    assert unreadable == []
    assert sorted(warned) == sorted(WARNED_EXAMPLES)
    for key, words in WARNED_EXAMPLES.items():
        assert len(warned[key]) == len(words), warned[key]
        for word, warning in zip(words, warned[key]):
            assert word in warning
