# ruff: noqa: S101
"""Checks for the example folder loader."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tool.examples import (
    ExampleError,
    ExampleFiles,
    ExpectedComponent,
    ExpectedDataset,
    ExpectedScalar,
    example_dir,
    load_example,
    read_csv_rows,
    scan_example_dirs,
)

SCRIPT = "DS_r := DS_1 * 2;\n"
TYPE_MESSAGE = "the type of Me_1 must be a string or a non-empty list of strings"
SCALAR_MESSAGE = "the value of sc must be a string, a number, a boolean or null"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def component(name: str, role: str, type_: Any, **extra: Any) -> dict[str, Any]:
    return {"name": name, "role": role, "type": type_, **extra}


def structure(name: str, *components: Any) -> str:
    """The text of a structure file that describes one dataset."""
    return json.dumps({"datasets": [{"name": name, "DataStructure": list(components)}]})


def scalars(*items: dict[str, Any]) -> str:
    """The text of an ``expected/scalars.json`` file."""
    return json.dumps({"scalars": list(items)})


def scalar(name: str, type_: str, value: Any) -> dict[str, Any]:
    return {"name": name, "type": type_, "value": value}


def build_example(path: Path) -> Path:
    """``DS_r := DS_1 * 2`` with its input ``DS_1`` and its expected result ``DS_r``."""
    write(path / "script.vtl", SCRIPT)
    write(
        path / "input/DS_1.json",
        structure(
            "DS_1",
            component("Id_1", "Identifier", "Integer", nullable=False),
            component("Me_1", "Measure", "Integer"),
            component("At_1", "ViralAttribute", "String", nullable=True),
        ),
    )
    write(path / "input/DS_1.csv", "Id_1,Me_1,At_1\n1,10,a\n2,,b\n")
    write(
        path / "expected/DS_r.json",
        structure(
            "DS_r",
            component("Id_1", "Identifier", "Integer", nullable=False),
            component("Me_1", "Measure", ["Integer", "Number"]),
            component("At_1", "ViralAttribute", "String", nullable=True),
        ),
    )
    write(path / "expected/DS_r.csv", "Id_1,Me_1,At_1\n1,20,a\n2,,b\n")
    return path


def build_scalar_example(path: Path) -> Path:
    """``sc_r := 3 + 5`` has no input at all, only an expected scalar."""
    write(path / "script.vtl", "sc_r := 3 + 5;\n")
    write(path / "expected/scalars.json", scalars(scalar("sc_r", "Integer", 8)))
    return path


def only_dataset(files: ExampleFiles) -> ExpectedDataset:
    assert len(files.expected_datasets) == 1
    return files.expected_datasets[0]


# --- Layout and loading ---


def test_example_dir() -> None:
    root = Path("examples")
    assert example_dir(root, "numeric", "addition", "ds_ds", "minimal") == (
        root / "numeric" / "addition" / "ds_ds" / "minimal"
    )


def test_load_minimal_example(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    files = load_example(path)

    assert files.path == path
    assert files.script == SCRIPT
    assert files.structures == [path / "input/DS_1.json"]
    assert files.datapoints == {"DS_1": path / "input/DS_1.csv"}
    assert files.value_domains == []
    assert files.external_routines == []
    assert files.expected_scalars == []

    assert files.expected_datasets[0].components["Me_1"].types == ("Integer", "Number")
    dataset = only_dataset(files)
    assert dataset.name == "DS_r"
    # Roles are kept as written, a missing ``nullable`` is None, and the JSON order is kept.
    assert dataset.components == {
        "Id_1": ExpectedComponent("Identifier", ("Integer",), False),
        "Me_1": ExpectedComponent("Measure", ("Integer", "Number"), None),
        "At_1": ExpectedComponent("ViralAttribute", ("String",), True),
    }
    assert list(dataset.components) == ["Id_1", "Me_1", "At_1"]
    assert dataset.rows == [
        {"Id_1": "1", "Me_1": "20", "At_1": "a"},
        {"Id_1": "2", "Me_1": None, "At_1": "b"},
    ]


def test_input_files_are_listed_by_name_and_only_csvs_become_datapoints(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    # DS_0 has a structure but no data: the dataset is empty and has no datapoints entry.
    write(path / "input/DS_0.json", structure("DS_0", component("Id_1", "Identifier", "String")))
    # Created in an order that is neither the sorted one nor its reverse.
    for name in ("d", "b", "e", "a", "c"):
        write(path / f"value_domains/{name}.json", "{}")
    write(path / "value_domains/readme.txt", "not a value domain")
    (path / "value_domains/folder.json").mkdir()
    write(path / "external_routines/r.json", "{}")

    files = load_example(path)

    assert files.structures == [path / "input/DS_0.json", path / "input/DS_1.json"]
    assert files.datapoints == {"DS_1": path / "input/DS_1.csv"}
    assert files.value_domains == [path / f"value_domains/{name}.json" for name in "abcde"]
    assert files.external_routines == [path / "external_routines/r.json"]


def test_expected_datasets_are_sorted_by_file_name(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    write(path / "expected/DS_s.json", structure("DS_s", component("Id_1", "Identifier", "String")))
    write(path / "expected/DS_s.csv", "Id_1\nx\n")
    write(path / "expected/DS_q.json", structure("DS_q", component("Id_1", "Identifier", "String")))
    write(path / "expected/DS_q.csv", "Id_1\n")

    files = load_example(path)

    assert [dataset.name for dataset in files.expected_datasets] == ["DS_q", "DS_r", "DS_s"]
    assert [len(dataset.rows) for dataset in files.expected_datasets] == [0, 2, 1]


def test_scalar_only_example_has_no_inputs(tmp_path: Path) -> None:
    files = load_example(build_scalar_example(tmp_path / "ex"))
    assert files.script == "sc_r := 3 + 5;\n"
    assert files.structures == []
    assert files.datapoints == {}
    assert files.expected_datasets == []
    assert files.expected_scalars == [ExpectedScalar("sc_r", "Integer", "8")]


def test_scalar_values_keep_their_written_form(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    write(
        path / "expected/scalars.json",
        """{"scalars": [
            {"name": "sc_1", "type": "Integer", "value": 8},
            {"name": "sc_2", "type": "Number", "value": 10.50},
            {"name": "sc_3", "type": "Number", "value": 1E-7},
            {"name": "sc_4", "type": "String", "value": " a b "},
            {"name": "sc_5", "type": "Boolean", "value": true},
            {"name": "sc_6", "type": "Boolean", "value": false},
            {"name": "sc_7", "type": "Integer", "value": null},
            {"name": "sc_8", "type": "String", "value": ""}
        ]}""",
    )

    files = load_example(path)

    # A dataset output and scalar outputs can live in the same example.
    assert only_dataset(files).name == "DS_r"
    assert files.expected_scalars == [
        ExpectedScalar("sc_1", "Integer", "8"),
        ExpectedScalar("sc_2", "Number", "10.50"),  # the printed precision matters
        ExpectedScalar("sc_3", "Number", "1E-7"),
        ExpectedScalar("sc_4", "String", " a b "),
        ExpectedScalar("sc_5", "Boolean", "true"),
        ExpectedScalar("sc_6", "Boolean", "false"),
        ExpectedScalar("sc_7", "Integer", None),  # null is written null...
        ExpectedScalar("sc_8", "String", ""),  # ...and the empty string stays a string
    ]


def test_empty_expected_result_is_a_header_only_csv(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    write(path / "expected/DS_r.csv", "Id_1,Me_1,At_1\n")
    assert only_dataset(load_example(path)).rows == []


def test_utf8_bom_is_ignored_in_every_file(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    for file in ("script.vtl", "expected/DS_r.json", "expected/DS_r.csv", "input/DS_1.csv"):
        text = (path / file).read_text(encoding="utf-8")
        (path / file).write_text("\ufeff" + text, encoding="utf-8")
    files = load_example(path)
    assert files.script == SCRIPT
    assert list(only_dataset(files).components) == ["Id_1", "Me_1", "At_1"]
    assert only_dataset(files).rows[0] == {"Id_1": "1", "Me_1": "20", "At_1": "a"}


def test_loaded_values_are_frozen_dataclasses(tmp_path: Path) -> None:
    files = load_example(build_example(tmp_path / "ex"))
    dataset = only_dataset(files)
    for value, field in (
        (files, "script"),
        (dataset, "name"),
        (dataset.components["Id_1"], "role"),
        (ExpectedScalar("sc", "Integer", "1"), "value"),
    ):
        assert dataclasses.is_dataclass(value)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, "other")


# --- CSV files ---


def test_csv_bom_quotes_and_nulls(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text('\ufeffId_1,Me_1\n1,"a,b"\n2,\n', encoding="utf-8")
    assert read_csv_rows(p) == [{"Id_1": "1", "Me_1": "a,b"}, {"Id_1": "2", "Me_1": None}]


def test_read_csv_rows_keeps_values_verbatim(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    # CRLF lines, a blank line, padded and quoted values, a multi-line cell and no final newline.
    p.write_bytes(b'Id_1,Me_1\r\n1, padded \r\n\r\n2,"say ""hi""\r\nthere"\r\n3,"  "')
    assert read_csv_rows(p) == [
        {"Id_1": "1", "Me_1": " padded "},
        {"Id_1": "2", "Me_1": 'say "hi"\r\nthere'},
        {"Id_1": "3", "Me_1": "  "},
    ]
    p.write_text("Id_1,Me_1\n", encoding="utf-8")
    assert read_csv_rows(p) == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Id_1,Me_1\n1,10\n2\n", "line 3 has fewer fields than the header"),
        ("Id_1,Me_1\n1,10,11\n", "line 2 has more fields than the header"),
        ("Id_1,Me_1,Id_1\n1,10,11\n", "duplicate column 'Id_1'"),
    ],
)
def test_read_csv_rows_rejects_malformed_files(tmp_path: Path, text: str, message: str) -> None:
    p = tmp_path / "data.csv"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ExampleError, match=message):
        read_csv_rows(p)


def test_read_csv_rows_rejects_files_that_are_not_utf8(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_bytes(b"Id_1,Me_1\n1,caf\xe9\n")
    with pytest.raises(ExampleError, match="not valid UTF-8"):
        read_csv_rows(p)


# --- Missing and inconsistent files ---


def test_missing_script_or_expected_raises(tmp_path: Path) -> None:
    no_script = build_example(tmp_path / "no_script")
    (no_script / "script.vtl").unlink()
    with pytest.raises(ExampleError, match=r"missing script\.vtl") as info:
        load_example(no_script)
    assert str(no_script) in str(info.value)  # the message names the example folder

    no_expected = build_example(tmp_path / "no_expected")
    shutil.rmtree(no_expected / "expected")
    with pytest.raises(ExampleError, match=r"missing expected/ folder"):
        load_example(no_expected)

    # A folder that does not exist at all has no script either.
    with pytest.raises(ExampleError, match=r"missing script\.vtl"):
        load_example(tmp_path / "nope")


def test_example_needs_an_expected_output(tmp_path: Path) -> None:
    path = build_scalar_example(tmp_path / "ex")
    write(path / "expected/scalars.json", scalars())
    with pytest.raises(ExampleError, match="no expected output"):
        load_example(path)

    (path / "expected/scalars.json").unlink()
    with pytest.raises(ExampleError, match="no expected output"):
        load_example(path)


def test_expected_dataset_needs_its_json_and_csv(tmp_path: Path) -> None:
    no_csv = build_example(tmp_path / "no_csv")
    (no_csv / "expected/DS_r.csv").unlink()
    with pytest.raises(ExampleError, match=r"missing expected/DS_r\.csv"):
        load_example(no_csv)

    no_json = build_example(tmp_path / "no_json")
    (no_json / "expected/DS_r.json").unlink()
    with pytest.raises(ExampleError, match=r"missing expected/DS_r\.json"):
        load_example(no_json)

    # The pair rule also holds next to a scalar expectation.
    with_scalar = build_scalar_example(tmp_path / "with_scalar")
    write(with_scalar / "expected/DS_r.csv", "Id_1\n1\n")
    with pytest.raises(ExampleError, match=r"missing expected/DS_r\.json"):
        load_example(with_scalar)


def test_expected_json_must_hold_one_dataset_named_like_its_file(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    entries = [{"name": "DS_r", "DataStructure": []}, {"name": "DS_s", "DataStructure": []}]
    write(path / "expected/DS_r.json", json.dumps({"datasets": entries}))
    with pytest.raises(ExampleError, match="exactly one dataset, found 2"):
        load_example(path)

    write(path / "expected/DS_r.json", json.dumps({"datasets": []}))
    with pytest.raises(ExampleError, match="exactly one dataset, found 0"):
        load_example(path)

    write(path / "expected/DS_r.json", structure("DS_x", component("Id_1", "Identifier", "String")))
    with pytest.raises(ExampleError, match="the dataset is named 'DS_x', not 'DS_r'"):
        load_example(path)


def test_expected_csv_columns_must_match_the_structure(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    write(path / "expected/DS_r.csv", "Id_1,Me_9,At_1\n1,20,a\n")
    with pytest.raises(
        ExampleError,
        match=r"columns do not match DS_r\.json \(missing Me_1; not in the structure: Me_9\)",
    ):
        load_example(path)

    # A file without even a header row is not the way to write an empty result.
    write(path / "expected/DS_r.csv", "")
    with pytest.raises(ExampleError, match=r"\(missing Id_1, Me_1, At_1\)"):
        load_example(path)


def test_expected_output_names_must_be_unique(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    write(path / "expected/scalars.json", scalars(scalar("DS_r", "Integer", 1)))
    with pytest.raises(ExampleError, match="expected output 'DS_r' is defined twice"):
        load_example(path)

    write(
        path / "expected/scalars.json",
        scalars(scalar("a", "Integer", 1), scalar("a", "Integer", 2)),
    )
    with pytest.raises(ExampleError, match="expected output 'a' is defined twice"):
        load_example(path)


MALFORMED_FILES = [
    ("expected/DS_r.json", "{not json", "invalid JSON"),
    ("expected/DS_r.json", "[]", "the document must be an object"),
    ("expected/DS_r.json", '{"datasets": 3}', "'datasets' must be an array"),
    ("expected/DS_r.json", '{"datasets": [3]}', "the dataset must be an object"),
    ("expected/DS_r.json", '{"datasets": [{"name": 3}]}', "the dataset name must be a string"),
    (
        "expected/DS_r.json",
        '{"datasets": [{"name": "DS_r"}]}',
        "'DataStructure' must be an array",
    ),
    ("expected/DS_r.json", structure("DS_r", 3), "a component must be an object"),
    (
        "expected/DS_r.json",
        structure("DS_r", {"role": "Measure", "type": "Integer"}),
        "a component name must be a string",
    ),
    (
        "expected/DS_r.json",
        structure("DS_r", {"name": "Me_1", "type": "Integer"}),
        "the role of Me_1 must be a string",
    ),
    (
        "expected/DS_r.json",
        structure("DS_r", {"name": "Me_1", "role": "Measure"}),
        TYPE_MESSAGE,
    ),
    ("expected/DS_r.json", structure("DS_r", component("Me_1", "Measure", [])), TYPE_MESSAGE),
    (
        "expected/DS_r.json",
        structure("DS_r", component("Me_1", "Measure", ["Integer", 3])),
        TYPE_MESSAGE,
    ),
    (
        "expected/DS_r.json",
        structure("DS_r", component("Me_1", "Measure", "Integer", nullable="no")),
        "nullable of Me_1 must be true or false",
    ),
    (
        "expected/DS_r.json",
        structure("DS_r", component("Me_1", "Measure", "Integer", nullable=1)),
        "nullable of Me_1 must be true or false",
    ),
    (
        "expected/DS_r.json",
        structure(
            "DS_r",
            component("Me_1", "Measure", "Integer"),
            component("Me_1", "Measure", "Number"),
        ),
        "duplicate component 'Me_1'",
    ),
    ("expected/scalars.json", "{not json", "invalid JSON"),
    ("expected/scalars.json", '{"scalars": {}}', "'scalars' must be an array"),
    ("expected/scalars.json", '{"scalars": [3]}', "a scalar must be an object"),
    (
        "expected/scalars.json",
        scalars({"type": "Integer", "value": 1}),
        "a scalar name must be a string",
    ),
    (
        "expected/scalars.json",
        scalars({"name": "sc", "value": 1}),
        "the type of sc must be a string",
    ),
    (
        "expected/scalars.json",
        scalars({"name": "sc", "type": "Integer"}),
        "scalar 'sc' has no value",
    ),
    ("expected/scalars.json", scalars(scalar("sc", "Integer", [1])), SCALAR_MESSAGE),
    ("expected/scalars.json", scalars(scalar("sc", "Integer", {"a": 1})), SCALAR_MESSAGE),
]


@pytest.mark.parametrize(
    ("file", "content", "message"),
    MALFORMED_FILES,
    ids=[message for _, _, message in MALFORMED_FILES],
)
def test_malformed_files_raise_example_error(
    tmp_path: Path, file: str, content: str, message: str
) -> None:
    path = build_example(tmp_path / "ex")
    write(path / file, content)
    with pytest.raises(ExampleError, match=message) as info:
        load_example(path)
    assert str(path / file) in str(info.value)  # the message names the offending file


def test_files_that_are_not_utf8_raise_example_error(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    (path / "script.vtl").write_bytes(b"DS_r := DS_1 * 2; /* caf\xe9 */\n")
    with pytest.raises(ExampleError, match="not valid UTF-8"):
        load_example(path)


# --- Scanning a folder of examples ---


def test_scan_example_dirs(tmp_path: Path) -> None:
    for name in ("manual_ex_1", "minimal"):
        build_example(example_dir(tmp_path, "numeric", "addition", "ds_ds", name))
    assert scan_example_dirs(tmp_path) == {
        ("numeric", "addition", "ds_ds", "manual_ex_1"),
        ("numeric", "addition", "ds_ds", "minimal"),
    }


def test_scan_example_dirs_ignores_other_depths_and_folders_without_script(
    tmp_path: Path,
) -> None:
    write(tmp_path / "chapter/operator/variant/script.vtl", "")  # one level too shallow
    write(tmp_path / "chapter/operator/variant/example/deeper/script.vtl", "")  # one too deep
    write(tmp_path / "script.vtl", "")
    write(tmp_path / "chapter/operator/variant/example/notes.md", "no script here")
    write(tmp_path / "chapter/operator/variant/example/other.vtl", "only script.vtl counts")
    (tmp_path / "chapter/operator/variant/empty").mkdir()
    (tmp_path / "chapter/operator/variant/folder/script.vtl").mkdir(parents=True)  # not a file
    write(tmp_path / "types/number/identity/minimal/script.vtl", "")  # the types chapter

    assert scan_example_dirs(tmp_path) == {("types", "number", "identity", "minimal")}
    assert scan_example_dirs(tmp_path / "does_not_exist") == set()
