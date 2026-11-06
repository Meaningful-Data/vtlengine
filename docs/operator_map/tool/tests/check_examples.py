# ruff: noqa: S101
"""Checks for the example folder loader."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tool.examples import (
    ERROR_CODE,
    ExampleError,
    ExampleFiles,
    ExpectedComponent,
    ExpectedDataset,
    ExpectedScalar,
    ScalarValue,
    check_notes,
    example_dir,
    load_example,
    notes_problems,
    read_csv_rows,
    scan_example_dirs,
)
from tool.model import Example, authored, manual

SCRIPT = "DS_r := DS_1 * 2;\n"
TYPE_MESSAGE = "the type of Me_1 must be a string or a non-empty list of strings"
SCALAR_MESSAGE = "the value of sc must be a string, a number, a boolean or null"
ERROR_MESSAGE = "the code must be a string"


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


def declared(*pairs: tuple[str, str]) -> str:
    """The text of an ``input/scalars.json`` file: the input scalars, each with its type."""
    return json.dumps({"scalars": [{"name": name, "type": type_} for name, type_ in pairs]})


def row(*cells: str) -> str:
    """One line of CSV, a cell quoted where it needs it."""
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerow(cells)
    return out.getvalue()


def build_scalar_input_example(path: Path) -> Path:
    """``sc_r := sc_1 + sc_2`` reads its two operands from ``input/scalars.csv``."""
    write(path / "script.vtl", "sc_r := sc_1 + sc_2;\n")
    write(path / "input/scalars.json", declared(("sc_1", "Integer"), ("sc_2", "Integer")))
    write(path / "input/scalars.csv", row("sc_1", "sc_2") + row("9", "4"))
    write(path / "expected/scalars.json", scalars(scalar("sc_r", "Integer", 13)))
    return path


def write_scalar_input(path: Path, type_: str, cell: str) -> None:
    """The input scalar ``sc_1`` of ``type_`` that holds ``cell`` in ``input/scalars.csv``."""
    write(path / "input/scalars.json", declared(("sc_1", type_)))
    write(path / "input/scalars.csv", row("sc_1") + row(cell))


def input_scalars(path: Path) -> dict[str, ScalarValue]:
    """The values of the input scalars of the example ``path``, which has some."""
    values = load_example(path).scalar_values
    assert values is not None
    return values


def error_file(code: Any) -> str:
    """The text of an ``expected/error.json`` file."""
    return json.dumps({"code": code})


def build_error_example(path: Path) -> Path:
    """``DS_r := DS_1 / 0`` has an input and expects an error, so it has no expected output."""
    write(path / "script.vtl", "DS_r := DS_1 / 0;\n")
    write(
        path / "input/DS_1.json",
        structure(
            "DS_1",
            component("Id_1", "Identifier", "Integer", nullable=False),
            component("Me_1", "Measure", "Integer"),
        ),
    )
    write(path / "input/DS_1.csv", "Id_1,Me_1\n1,10\n")
    write(path / "expected/error.json", error_file("2-1-15-6"))
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
    assert files.expected_error is None

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


def test_input_csv_needs_a_structure_that_declares_its_dataset(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    ids = [component("Id_1", "Identifier", "Integer")]

    def declare(file: str, *names: str) -> None:
        datasets = [{"name": name, "DataStructure": ids} for name in names]
        write(path / f"input/{file}.json", json.dumps({"datasets": datasets}))

    # A structure file may declare several datasets, in any order, so DS_4 and DS_3 need no file
    # of their own. A structure file that declares scalars only, no dataset, is accepted too.
    declare("DS_1", "DS_4", "DS_1")
    declare("DS_2", "DS_3", "DS_2")
    write(path / "input/sc.json", '{"scalars": [{"name": "sc_1", "type": "Integer"}]}')
    for name in ("DS_2", "DS_3", "DS_4"):
        write(path / f"input/{name}.csv", "Id_1\n1\n")
    assert list(load_example(path).datapoints) == ["DS_1", "DS_2", "DS_3", "DS_4"]

    # A typo in a file name: no structure declares DS_l, and run() would refuse the file.
    write(path / "input/DS_l.csv", "Id_1\n1\n")
    message = r"input/DS_l\.csv: no structure declares the dataset 'DS_l' "
    declared = r"\(declared in input/\*\.json: DS_1, DS_2, DS_3, DS_4\)"
    with pytest.raises(ExampleError, match=message + declared):
        load_example(path)

    # Without any structure file nothing is declared.
    for file in list((path / "input").glob("*.json")):
        file.unlink()
    message = r"input/DS_1\.csv: no structure declares the dataset 'DS_1' "
    with pytest.raises(ExampleError, match=message + r"\(declared in input/\*\.json: none\)"):
        load_example(path)


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


# --- Expected errors ---


def test_error_json_loads(tmp_path: Path) -> None:
    path = build_error_example(tmp_path / "ex")
    files = load_example(path)

    assert files.expected_error == "2-1-15-6"
    # The error is the expected result: no expected output goes with it. ``error.json`` is not
    # taken for the structure of a dataset either, which would want an ``error.csv``.
    assert files.expected_datasets == []
    assert files.expected_scalars == []
    # The rest of the example is read as in any other.
    assert files.script == "DS_r := DS_1 / 0;\n"
    assert files.structures == [path / "input/DS_1.json"]
    assert files.datapoints == {"DS_1": path / "input/DS_1.csv"}


@pytest.mark.parametrize(
    ("text", "is_a_code"),
    [
        ("2-1-15-6", True),
        ("1-2-23", True),
        ("0-1", True),
        ("x", False),
        ("", False),
        ("2", False),
        ("2-", False),
        ("-2-1", False),
        ("2-1-a", False),
        (" 2-1-15-6", False),
        ("2-1-15-6\n", False),  # the whole text is the code
        ("\uff12-\uff11", False),  # fullwidth digits are not ASCII digits
    ],
)
def test_error_code_is_digits_in_two_groups_or_more(text: str, is_a_code: bool) -> None:
    assert (ERROR_CODE.fullmatch(text) is not None) is is_a_code


def test_an_example_that_expects_outputs_expects_no_error(tmp_path: Path) -> None:
    assert load_example(build_example(tmp_path / "dataset")).expected_error is None
    assert load_example(build_scalar_example(tmp_path / "scalar")).expected_error is None


@pytest.mark.parametrize("code", ["2-1-15-6", "1-1-1-1", "0-5-1-9", "1-2-23", "2-3-6", "0-1"])
def test_error_json_takes_a_code_of_the_form_of_those_of_vtlengine(
    tmp_path: Path, code: str
) -> None:
    path = build_error_example(tmp_path / "ex")
    write(path / "expected/error.json", error_file(code))
    assert load_example(path).expected_error == code


def test_error_json_may_start_with_a_byte_order_mark(tmp_path: Path) -> None:
    path = build_error_example(tmp_path / "ex")
    (path / "expected/error.json").write_text("\ufeff" + error_file("2-1-15-6"), encoding="utf-8")
    assert load_example(path).expected_error == "2-1-15-6"


def test_error_json_with_outputs_is_rejected(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    write(path / "expected/error.json", error_file("2-1-15-6"))
    with pytest.raises(ExampleError) as info:
        load_example(path)
    message = str(info.value)
    # The message names the file that is out of place, and the outputs that it cannot go with.
    assert str(path / "expected/error.json") in message
    assert "expected/DS_r.json" in message

    # Scalars are outputs too, and so is each file of an output that there is.
    write(path / "expected/scalars.json", scalars(scalar("sc", "Integer", 1)))
    with pytest.raises(ExampleError) as info:
        load_example(path)
    assert "expected/DS_r.json, expected/scalars.json" in str(info.value)

    only_scalars = build_scalar_example(tmp_path / "scalars")
    write(only_scalars / "expected/error.json", error_file("2-1-15-6"))
    with pytest.raises(ExampleError) as info:
        load_example(only_scalars)
    assert "expected/scalars.json" in str(info.value)
    assert str(only_scalars / "expected/error.json") in str(info.value)


@pytest.mark.parametrize(
    "code",
    # The last one is written with the fullwidth digits U+FF12 and U+FF11: not ASCII digits.
    [
        "x",
        "",
        "2",
        "2-",
        "-2-1",
        "2-1-",
        "2-1-a",
        " 2-1-15-6",
        "2-1-15-6 ",
        "2.1.15.6",
        "\uff12-\uff11",
    ],
)
def test_error_json_bad_code_is_rejected(tmp_path: Path, code: str) -> None:
    path = build_error_example(tmp_path / "ex")
    write(path / "expected/error.json", error_file(code))
    with pytest.raises(ExampleError, match="not the code of a vtlengine error") as info:
        load_example(path)
    assert str(path / "expected/error.json") in str(info.value)
    assert repr(code) in str(info.value)  # the message shows what was written


def test_the_message_of_an_example_without_outputs_names_the_error_file(tmp_path: Path) -> None:
    path = build_error_example(tmp_path / "ex")
    (path / "expected/error.json").unlink()
    with pytest.raises(ExampleError, match="no expected output") as info:
        load_example(path)
    assert "expected/error.json" in str(info.value)


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


# --- Input scalars ---


def test_input_scalars_are_read_from_scalars_csv_typed_by_scalars_json(tmp_path: Path) -> None:
    path = build_scalar_input_example(tmp_path / "ex")
    files = load_example(path)

    assert files.scalar_values == {"sc_1": 9, "sc_2": 4}
    assert [type(value) for value in input_scalars(path).values()] == [int, int]
    # scalars.json declares them, and run() needs it: it stays among the structures. scalars.csv
    # is the values of the scalars, not the data points of a dataset.
    assert files.structures == [path / "input/scalars.json"]
    assert files.datapoints == {}
    assert files.expected_scalars == [ExpectedScalar("sc_r", "Integer", "13")]


def test_input_scalars_go_next_to_input_datasets(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    write(path / "input/scalars.json", declared(("sc_1", "Number"), ("sc_2", "String")))
    write(path / "input/scalars.csv", row("sc_1", "sc_2") + row("2", "a, b"))

    files = load_example(path)

    assert files.scalar_values == {"sc_1": 2.0, "sc_2": "a, b"}
    assert files.structures == [path / "input/DS_1.json", path / "input/scalars.json"]
    assert files.datapoints == {"DS_1": path / "input/DS_1.csv"}


def test_an_example_without_scalars_csv_has_no_scalar_values(tmp_path: Path) -> None:
    assert load_example(build_example(tmp_path / "dataset")).scalar_values is None
    assert load_example(build_scalar_example(tmp_path / "scalar")).scalar_values is None
    # A structure file that declares scalars, but is not scalars.json, gives them no value.
    path = build_example(tmp_path / "declared")
    write(path / "input/sc.json", declared(("sc_1", "Integer")))
    assert load_example(path).scalar_values is None


def test_scalar_values_are_in_the_order_of_scalars_json(tmp_path: Path) -> None:
    path = build_scalar_input_example(tmp_path / "ex")
    write(path / "input/scalars.csv", row("sc_2", "sc_1") + row("4", "9"))
    assert list(input_scalars(path).items()) == [("sc_1", 9), ("sc_2", 4)]


@pytest.mark.parametrize(
    ("type_", "cell", "value"),
    [
        ("Integer", "9", 9),
        ("Integer", "-3", -3),
        ("Integer", "+5", 5),
        ("Integer", "007", 7),
        ("Number", "9", 9.0),
        ("Number", "1.5", 1.5),
        ("Number", "-0.25", -0.25),
        ("Number", ".5", 0.5),
        ("Number", "5.", 5.0),
        ("Number", "1E-7", 1e-7),
        ("Number", "+2e3", 2000.0),
        ("Boolean", "true", True),
        ("Boolean", "false", False),
        ("Boolean", "TRUE", True),
        ("Boolean", "False", False),
        ("String", " a, b ", " a, b "),  # verbatim: the spaces stay
        ("String", "true", "true"),  # a String is not read as what it looks like
        ("String", "9", "9"),
        ("Date", "2020-01-31", "2020-01-31"),
        ("Time_Period", "2020Q1", "2020Q1"),
        ("Time", "2020-01-01/2020-12-31", "2020-01-01/2020-12-31"),
        ("Duration", "M", "M"),
    ],
)
def test_an_input_scalar_has_the_python_type_of_its_vtl_type(
    tmp_path: Path, type_: str, cell: str, value: Any
) -> None:
    path = build_scalar_input_example(tmp_path / "ex")
    write_scalar_input(path, type_, cell)

    (loaded,) = input_scalars(path).values()

    # 9 == 9.0 and True == 1 in Python, so the type is compared apart.
    assert (type(loaded), loaded) == (type(value), value)


@pytest.mark.parametrize(
    "type_",
    ["Integer", "Number", "Boolean", "String", "Date", "Time_Period", "Time", "Duration"],
)
def test_an_empty_cell_is_a_null_scalar_of_any_type(tmp_path: Path, type_: str) -> None:
    path = build_scalar_input_example(tmp_path / "ex")
    write_scalar_input(path, type_, "")
    assert load_example(path).scalar_values == {"sc_1": None}


def test_input_scalars_read_a_file_with_a_byte_order_mark(tmp_path: Path) -> None:
    path = build_scalar_input_example(tmp_path / "ex")
    for file in ("input/scalars.csv", "input/scalars.json"):
        text = (path / file).read_text(encoding="utf-8")
        (path / file).write_text("\ufeff" + text, encoding="utf-8")
    assert load_example(path).scalar_values == {"sc_1": 9, "sc_2": 4}


def test_scalars_csv_and_scalars_json_go_together(tmp_path: Path) -> None:
    no_structure = build_scalar_input_example(tmp_path / "no_structure")
    (no_structure / "input/scalars.json").unlink()
    with pytest.raises(ExampleError) as info:
        load_example(no_structure)
    assert str(info.value) == (
        f"{no_structure}: missing input/scalars.json (it goes with input/scalars.csv)"
    )

    no_values = build_scalar_input_example(tmp_path / "no_values")
    (no_values / "input/scalars.csv").unlink()
    with pytest.raises(ExampleError) as info:
        load_example(no_values)
    assert str(info.value) == (
        f"{no_values}: missing input/scalars.csv (it goes with input/scalars.json)"
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        # A name that scalars.json does not declare, a declared scalar with no column, and both.
        ("sc_1,sc_2,sc_9\n9,4,1\n", r"\(not in the structure: sc_9\)"),
        ("sc_1\n9\n", r"\(missing sc_2\)"),
        ("sc_1,sc_9\n9,1\n", r"\(missing sc_2; not in the structure: sc_9\)"),
        ("", r"\(missing sc_1, sc_2\)"),  # not even a header
    ],
)
def test_the_columns_of_scalars_csv_are_the_scalars_of_scalars_json(
    tmp_path: Path, text: str, message: str
) -> None:
    path = build_scalar_input_example(tmp_path / "ex")
    write(path / "input/scalars.csv", text)
    with pytest.raises(
        ExampleError, match=r"columns do not match scalars\.json " + message
    ) as info:
        load_example(path)
    assert str(path / "input/scalars.csv") in str(info.value)


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("sc_1,sc_2\n", 0),
        ("sc_1,sc_2\n9,4\n10,5\n", 2),
        ("sc_1,sc_2\n9,4\n10,5\n11,6\n", 3),
    ],
)
def test_scalars_csv_has_exactly_one_data_row(tmp_path: Path, text: str, found: int) -> None:
    path = build_scalar_input_example(tmp_path / "ex")
    write(path / "input/scalars.csv", text)
    with pytest.raises(ExampleError) as info:
        load_example(path)
    assert str(info.value) == (
        f"{path / 'input/scalars.csv'}: must have exactly one data row, found {found}"
    )


@pytest.mark.parametrize(
    ("type_", "cell", "wanted"),
    [
        ("Integer", "abc", "an Integer"),
        ("Integer", "9.0", "an Integer"),  # a whole number is written without a point
        ("Integer", "1e3", "an Integer"),
        ("Integer", " 9", "an Integer"),
        ("Integer", "9 ", "an Integer"),
        ("Integer", "1_0", "an Integer"),  # Python's int() reads this one, the file does not
        ("Integer", "\u0663", "an Integer"),  # an Arabic-Indic digit is not an ASCII one
        ("Integer", "--1", "an Integer"),
        ("Integer", "true", "an Integer"),
        ("Number", "abc", "a Number"),
        ("Number", "1,5", "a Number"),
        ("Number", "1_0.5", "a Number"),
        ("Number", "nan", "a Number"),
        ("Number", "inf", "a Number"),
        ("Number", "-Infinity", "a Number"),
        ("Number", "1e999", "a Number"),  # not a finite number once it is read
        ("Number", ".", "a Number"),
        ("Number", "1 2", "a Number"),
        ("Boolean", "yes", "true or false"),
        ("Boolean", "1", "true or false"),
        ("Boolean", "0", "true or false"),
        ("Boolean", "t", "true or false"),
        ("Boolean", " true", "true or false"),
    ],
)
def test_a_value_that_is_not_of_the_type_of_its_scalar_is_refused(
    tmp_path: Path, type_: str, cell: str, wanted: str
) -> None:
    path = build_scalar_input_example(tmp_path / "ex")
    write_scalar_input(path, type_, cell)
    with pytest.raises(ExampleError) as info:
        load_example(path)
    assert str(info.value) == (
        f"{path / 'input/scalars.csv'}: the value of sc_1 must be {wanted}, not {cell!r}"
    )


def test_the_first_value_that_is_not_of_its_type_is_the_one_reported(tmp_path: Path) -> None:
    path = build_scalar_input_example(tmp_path / "ex")
    write(path / "input/scalars.csv", row("sc_2", "sc_1") + row("x", "y"))
    with pytest.raises(ExampleError, match="the value of sc_1 must be an Integer, not 'y'"):
        load_example(path)


SCALARS_MALFORMED = [
    ("{not json", "invalid JSON"),
    ("[]", "the document must be an object"),
    ("{}", "'scalars' must be an array"),
    ('{"scalars": {}}', "'scalars' must be an array"),
    ('{"scalars": [3]}', "a scalar must be an object"),
    ('{"scalars": [{"type": "Integer"}]}', "a scalar name must be a string"),
    ('{"scalars": [{"name": "sc_1"}]}', "the type of sc_1 must be a string"),
    ('{"scalars": [{"name": "sc_1", "type": 3}]}', "the type of sc_1 must be a string"),
    (
        '{"scalars": [{"name": "sc_1", "type": "Integr"}, {"name": "sc_2", "type": "Integer"}]}',
        "the type of sc_1 must be one of Integer, Number, Boolean, String, Date, Time_Period, "
        "Time, Duration, not 'Integr'",
    ),
    (
        '{"scalars": [{"name": "sc_1", "type": "integer"}, {"name": "sc_2", "type": "Integer"}]}',
        "the type of sc_1 must be one of Integer, .*, not 'integer'",  # the case matters to run()
    ),
    (declared(("sc_1", "Integer"), ("sc_1", "Integer")), "duplicate scalar 'sc_1'"),
]


@pytest.mark.parametrize(
    ("content", "message"),
    SCALARS_MALFORMED,
    ids=[message for _, message in SCALARS_MALFORMED],
)
def test_a_scalars_json_that_is_malformed_raises_example_error(
    tmp_path: Path, content: str, message: str
) -> None:
    path = build_scalar_input_example(tmp_path / "ex")
    write(path / "input/scalars.json", content)
    with pytest.raises(ExampleError, match=message) as info:
        load_example(path)
    assert str(path / "input/scalars.json") in str(info.value)


def test_a_scalars_csv_that_is_malformed_raises_example_error(tmp_path: Path) -> None:
    path = build_scalar_input_example(tmp_path / "ragged")
    write(path / "input/scalars.csv", "sc_1,sc_2\n9,4,1\n")
    with pytest.raises(ExampleError, match="line 2 has more fields than the header"):
        load_example(path)

    twice = build_scalar_input_example(tmp_path / "twice")
    write(twice / "input/scalars.csv", "sc_1,sc_1\n9,4\n")
    with pytest.raises(ExampleError, match="duplicate column 'sc_1'"):
        load_example(twice)

    not_utf8 = build_scalar_input_example(tmp_path / "not_utf8")
    (not_utf8 / "input/scalars.csv").write_bytes(b"sc_1,sc_2\n9,caf\xe9\n")
    with pytest.raises(ExampleError, match="not valid UTF-8"):
        load_example(not_utf8)


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
    # An error.json that cannot be read is reported as such, before it is found to be out of
    # place next to the expected dataset that these examples have.
    ("expected/error.json", "{not json", "invalid JSON"),
    ("expected/error.json", "[]", "the document must be an object"),
    ("expected/error.json", "{}", ERROR_MESSAGE),
    ("expected/error.json", '{"code": 3}', ERROR_MESSAGE),
    ("expected/error.json", '{"code": null}', ERROR_MESSAGE),
    ("expected/error.json", '{"code": ["2-1-15-6"]}', ERROR_MESSAGE),
    # The datasets that the input structures declare are read to match the input CSVs.
    ("input/DS_1.json", "{not json", "invalid JSON"),
    ("input/DS_1.json", "[]", "the document must be an object"),
    ("input/DS_1.json", '{"datasets": 3}', "'datasets' must be an array"),
    ("input/DS_1.json", '{"datasets": [3]}', "a dataset must be an object"),
    ("input/DS_1.json", '{"datasets": [{}]}', "the dataset name must be a string"),
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
    write(tmp_path / "general/non_persistent_assignment/ds/integer_measure/script.vtl", "")

    assert scan_example_dirs(tmp_path) == {
        ("general", "non_persistent_assignment", "ds", "integer_measure")
    }
    assert scan_example_dirs(tmp_path / "does_not_exist") == set()


# --- Notes ---


# The fixed layout: one field per line, the three that every example has.
NOTES = (
    ":Tests: The remainder of a division by zero.\n"
    ':Manual: "It returns op1 if divisor op2 is 0" (Modulo, Behaviour).\n'
    ":Expected: ``mod(9, 0)`` is 9, the dividend.\n"
)
ZERO_DIVISOR = authored("zero_divisor")
LAYOUT = (
    "they must be Tests, Manual, Expected, then any of Finding, Erratum, Adaptation, "
    "Team decision, in this order and once each"
)


def fields(*names: str) -> str:
    """A ``notes.rst`` text with one field per name, in that order, each on one line."""
    return "".join(f":{name}: Some text about {name.lower()}.\n" for name in names)


@pytest.mark.parametrize(
    ("names", "example"),
    [
        (("Tests", "Manual", "Expected"), ZERO_DIVISOR),
        (("Tests", "Manual", "Expected", "Finding"), ZERO_DIVISOR),
        (("Tests", "Manual", "Expected", "Erratum"), manual("ex_1", erratum="E")),
        (("Tests", "Manual", "Expected", "Adaptation"), manual("ex_1", adaptation="A")),
        (("Tests", "Manual", "Expected", "Team decision"), authored("count", decision=1049)),
        (
            ("Tests", "Manual", "Expected", "Finding", "Erratum", "Adaptation", "Team decision"),
            manual("ex_2", erratum="E", adaptation="A", decision=1049),
        ),
    ],
)
def test_check_notes_takes_the_fixed_layout(names: tuple[str, ...], example: Example) -> None:
    assert check_notes(fields(*names), example) == []


def test_check_notes() -> None:
    assert check_notes(NOTES, ZERO_DIVISOR) == []
    # Tests, Manual and Expected, in this order, then any of the others, in their order: each
    # once.
    for names in (
        ("Manual", "Tests", "Expected"),
        ("Tests", "Manual"),
        ("Tests", "Expected", "Manual"),
        ("Tests", "Finding", "Manual", "Expected"),
        ("Tests", "Manual", "Expected", "Tests"),
        ("Tests", "Manual", "Expected", "Finding", "Finding"),
        ("Expected",),
    ):
        assert check_notes(fields(*names), ZERO_DIVISOR) == [
            f"the fields are {', '.join(names)}: {LAYOUT}"
        ], names
    both = manual("ex_1", erratum="E", adaptation="A")
    wrong = ("Tests", "Manual", "Expected", "Adaptation", "Erratum")
    assert check_notes(fields(*wrong), both) == [f"the fields are {', '.join(wrong)}: {LAYOUT}"]
    # A field that the layout does not have is named; the others are still in order.
    assert check_notes(fields("Tests", "Manual", "Expected", "Notes"), ZERO_DIVISOR) == [
        "unknown field 'Notes'"
    ]
    assert check_notes(fields("Tests", "Comment", "Manual", "Expected"), ZERO_DIVISOR) == [
        "unknown field 'Comment'"
    ]
    # A text with no field at all is none of the layout.
    assert check_notes("Some notes.\n", ZERO_DIVISOR) == [
        "line 1 is outside the fields: a field starts with ':<Name>: ' and goes on in "
        "indented lines",
        f"the fields are none: {LAYOUT}",
    ]
    assert check_notes("", ZERO_DIVISOR) == [f"the fields are none: {LAYOUT}"]


@pytest.mark.parametrize(
    ("field", "flagged", "attribute"),
    [
        ("Erratum", manual("ex_1", erratum="E"), "erratum"),
        ("Adaptation", manual("ex_1", adaptation="A"), "adaptation"),
        ("Team decision", authored("count", decision=1049), "decision"),
    ],
)
def test_check_notes_has_a_field_for_an_erratum_an_adaptation_and_a_decision(
    field: str, flagged: Example, attribute: str
) -> None:
    # The field is there when the example says so in the catalogue, and only then.
    assert check_notes(NOTES, flagged) == [f"no field '{field}', but the example sets {attribute}"]
    assert check_notes(NOTES + f":{field}: Why.\n", ZERO_DIVISOR) == [
        f"a field '{field}', but the example does not set {attribute}"
    ]
    assert check_notes(NOTES + f":{field}: Why.\n", flagged) == []


def test_check_notes_are_1200_characters_at_most() -> None:
    exact = NOTES[:-1] + "x" * (1200 - len(NOTES)) + "\n"
    assert len(exact) == 1200
    assert check_notes(exact, ZERO_DIVISOR) == []
    longer = exact[:-1] + "y\n"
    assert check_notes(longer, ZERO_DIVISOR) == ["1201 characters: the notes have 1200 at most"]


def test_check_notes_has_one_paragraph_per_field() -> None:
    # A field goes on in indented lines, and the fields may be apart.
    goes_on = (
        ":Tests: The remainder of a division\n"
        "   by zero.\n"
        "\n"
        ":Manual:\n"
        '   "It returns op1 if divisor op2 is 0".\n'
        "\n"
        "\n"
        ":Expected: 9.\n"
    )
    assert check_notes(goes_on, ZERO_DIVISOR) == []
    # A blank line inside a field makes a second paragraph.
    two = ":Tests: The remainder.\n\n   Of a division by zero.\n:Manual: M.\n:Expected: E.\n"
    assert check_notes(two, ZERO_DIVISOR) == [
        "line 3: a blank line inside the field 'Tests': a field is one paragraph"
    ]
    after_marker = ":Tests:\n\n   The remainder.\n:Manual: M.\n:Expected: E.\n"
    assert check_notes(after_marker, ZERO_DIVISOR) == [
        "line 3: a blank line inside the field 'Tests': a field is one paragraph"
    ]
    # A line that is not indented is not in the field: the field list ends there.
    unindented = ":Tests: The remainder\nof a division.\n:Manual: M.\n:Expected: E.\n"
    outside = "is outside the fields: a field starts with ':<Name>: ' and goes on in indented lines"
    assert check_notes(unindented, ZERO_DIVISOR) == [f"line 2 {outside}"]
    titled = "Notes\n=====\n\n" + NOTES
    assert check_notes(titled, ZERO_DIVISOR) == [f"line 1 {outside}", f"line 2 {outside}"]
    # A field with nothing in it says nothing.
    empty = ":Tests:\n:Manual: M.\n:Expected:   \n"
    assert check_notes(empty, ZERO_DIVISOR) == [
        "the field 'Tests' is empty",
        "the field 'Expected' is empty",
    ]


def test_check_notes_reads_the_fields_only() -> None:
    # The markup of the text is for docutils to check (``check_notes_render.py``), not for this.
    text = (
        ':Tests: "quotes" *stars* ``code`` `role` | pipe :: colons, and :field: inside.\n'
        ":Manual: A rule::\n"
        ":Expected: *unclosed.\n"
    )
    assert check_notes(text, ZERO_DIVISOR) == []


def test_notes_problems_of_a_folder(tmp_path: Path) -> None:
    folder = build_example(tmp_path / "zero_divisor")
    assert notes_problems(folder, ZERO_DIVISOR) == [f"{folder}: missing notes.rst"]
    write(folder / "notes.rst", NOTES)
    assert notes_problems(folder, ZERO_DIVISOR) == []
    # The notes of the first version of the map are replaced, not kept beside.
    write(folder / "notes.md", "Free text.\n")
    old = f"{folder / 'notes.md'}: the notes go in notes.rst, not in notes.md"
    assert notes_problems(folder, ZERO_DIVISOR) == [old]
    (folder / "notes.rst").unlink()
    assert notes_problems(folder, ZERO_DIVISOR) == [old, f"{folder}: missing notes.rst"]
    # The problems of the text name the file.
    write(folder / "notes.rst", fields("Tests", "Expected"))
    (folder / "notes.md").unlink()
    assert notes_problems(folder, ZERO_DIVISOR) == [
        f"{folder / 'notes.rst'}: the fields are Tests, Expected: {LAYOUT}"
    ]
    assert notes_problems(folder, manual("ex_1", erratum="E")) == [
        f"{folder / 'notes.rst'}: the fields are Tests, Expected: {LAYOUT}",
        f"{folder / 'notes.rst'}: no field 'Erratum', but the example sets erratum",
    ]


def test_notes_problems_of_a_file_that_is_not_text(tmp_path: Path) -> None:
    folder = build_example(tmp_path / "zero_divisor")
    (folder / "notes.rst").write_bytes(b":Tests: caf\xe9.\n")
    [problem] = notes_problems(folder, ZERO_DIVISOR)
    assert problem.startswith(f"{folder / 'notes.rst'}: not valid UTF-8")
    # A byte order mark is not part of the text.
    (folder / "notes.rst").write_bytes(b"\xef\xbb\xbf" + NOTES.encode())
    assert notes_problems(folder, ZERO_DIVISOR) == []
