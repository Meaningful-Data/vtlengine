"""Layout and loading of one example folder of the operator map.

An example lives in ``examples/<chapter>/<operator>/<variant>/<example>/`` and holds:

* ``script.vtl``: the VTL script (required);
* ``input/<NAME>.json``: the structure of an input dataset, in the vtlengine format
  ``{"datasets": [{"name": ..., "DataStructure": [...]}]}``, and ``input/<NAME>.csv``: its data
  points (optional: without it the dataset is empty). The CSV of a dataset needs some structure
  file that declares a dataset ``NAME``, not necessarily ``input/<NAME>.json``;
* ``input/scalars.json`` and ``input/scalars.csv``: the input scalars, which go together. The JSON
  declares them, ``{"scalars": [{"name": ..., "type": ...}]}``, and the CSV holds their values: a
  column for each scalar, named after it, and exactly one data row, where an empty cell is a
  null. ``scalars.csv`` is not the data points of a dataset. The JSON types the values: an
  Integer is an ``int``, a Number a ``float``, a Boolean a ``bool`` (``true`` or ``false``, in any
  case), and a value of any other type is its text. A structure file with another name may
  declare scalars too, but then they have no value;
* ``value_domains/*.json`` and ``external_routines/*.json``: optional files for ``run``;
* ``expected/<NAME>.json`` and ``expected/<NAME>.csv``: an expected result dataset. The JSON has
  the same format, but a component ``type`` may also be a list of the allowed types. An empty
  result is a header-only CSV;
* ``expected/scalars.json``: the expected scalar results,
  ``{"scalars": [{"name": ..., "type": ..., "value": ...}]}``;
* ``expected/error.json``: the error that the script is to raise, ``{"code": "2-1-15-6"}``, with
  the code of a vtlengine error. It stands for the expected outputs: an example has the expected
  outputs or the expected error, not both;
* ``notes.rst``: the notes of the example, a field list in a fixed layout that ``check_notes``
  checks: what the example tests, the rule of the manual, the expected result, and a finding, an
  erratum, an adaptation or a team decision when one applies. The pages include it. It replaces
  ``notes.md``, the free text of the first version of the map, which the tool does not read.

An expected output or an expected error is required. Any problem of an example folder raises
``ExampleError``. The notes are not part of what ``load_example`` loads: the generator asks for
them, with ``notes_problems``.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeVar, Union

from tool.model import Example

_SCRIPT: Final = "script.vtl"
_INPUT: Final = "input"
_EXPECTED: Final = "expected"
_SCALARS: Final = "scalars.json"
_SCALAR_VALUES: Final = "scalars.csv"  # in ``input/``: the values of the input scalars
_ERROR: Final = "error.json"
_VALUE_DOMAINS: Final = "value_domains"
_EXTERNAL_ROUTINES: Final = "external_routines"
_NOTES: Final = "notes.rst"
_OLD_NOTES: Final = "notes.md"

# The fields of the notes, in their order: the first three always, then any of the others.
_REQUIRED_FIELDS: Final = ("Tests", "Manual", "Expected")
_OPTIONAL_FIELDS: Final = ("Finding", "Erratum", "Adaptation", "Team decision")
_LAYOUT: Final = (
    "they must be Tests, Manual, Expected, then any of Finding, Erratum, Adaptation, "
    "Team decision, in this order and once each"
)
# The field that an example has when the catalogue sets an attribute of it, and only then.
_FLAGGED_FIELDS: Final = (
    ("Erratum", "erratum"),
    ("Adaptation", "adaptation"),
    ("Team decision", "decision"),
)
_MAX_NOTES: Final = 1200  # characters
# The first line of a field: ``:<Name>:`` at the start of a line, then a space or the line end.
_FIELD: Final = re.compile(r"^:([A-Za-z ]+):\s")
_OUTSIDE: Final = (
    "is outside the fields: a field starts with ':<Name>: ' and goes on in indented lines"
)

# The code of a vtlengine error, such as ``1-1-1-1`` or ``2-3-6``: digits in two groups or more.
# It is written once, here: the loader checks the code of ``expected/error.json`` with it, and the
# child of the runner takes the code of an exception with it.
ERROR_CODE: Final = re.compile(r"\d+(-\d+)+", re.ASCII)

# The types that a structure file gives a scalar: the ``data-type`` of the JSON schema of vtlengine.
_SCALAR_TYPES: Final = (
    "Integer",
    "Number",
    "Boolean",
    "String",
    "Date",
    "Time_Period",
    "Time",
    "Duration",
)
# An integer or a number as a file writes it: ASCII digits only. ``int`` and ``float`` also take
# ``1_0``, ``nan``, ``inf`` and the digits of other scripts, and none of those is a value here.
_INTEGER: Final = re.compile(r"[+-]?\d+", re.ASCII)
_NUMBER: Final = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)

# The value of an input scalar: what ``run`` takes in ``scalar_values``. ``Union`` rather than
# ``|``: the alias is evaluated when the module is imported, on Python 3.9.
ScalarValue = Union[int, float, bool, str, None]

_T = TypeVar("_T")
_KINDS: Final[dict[type, str]] = {
    dict: "an object",
    list: "an array",
    str: "a string",
    bool: "true or false",
}


@dataclass(frozen=True)
class ExpectedComponent:
    """A component of an expected dataset."""

    role: str  # as written in the file: ``Identifier``, ``Measure``, ``ViralAttribute``...
    types: tuple[str, ...]  # the allowed VTL types: one, or several when the manual allows it
    nullable: bool | None  # None when the structure does not say


@dataclass(frozen=True)
class ExpectedDataset:
    """An expected result dataset: ``expected/<name>.json`` and ``expected/<name>.csv``."""

    name: str
    components: dict[str, ExpectedComponent]  # in file order
    rows: list[dict[str, str | None]]  # one dict per data point; an empty field is None


@dataclass(frozen=True)
class ExpectedScalar:
    """An expected scalar result, an item of ``expected/scalars.json``."""

    name: str
    type: str
    value: str | None  # as written in the file (``10.50`` stays ``"10.50"``); None for null


@dataclass(frozen=True)
class ExampleFiles:
    """What the tool needs to run one example and to judge its outputs."""

    path: Path
    script: str
    structures: list[Path]  # every ``input/*.json``, sorted by name
    # dataset name -> ``input/<name>.csv``, only datasets with data: ``scalars.csv`` is not one
    datapoints: dict[str, Path]
    value_domains: list[Path]  # ``value_domains/*.json``, sorted by name
    external_routines: list[Path]  # ``external_routines/*.json``, sorted by name
    expected_datasets: list[ExpectedDataset]  # sorted by file name
    expected_scalars: list[ExpectedScalar]  # in file order
    # The code of the error that the script is to raise, from ``expected/error.json``. An example
    # that has it has no expected outputs, and one that has none of the two cannot be loaded.
    expected_error: str | None = None
    # The values of the input scalars, by name, from ``input/scalars.csv``: typed as
    # ``input/scalars.json`` says (see ``_input_value``), and None for an empty cell. None itself
    # when the example has no input scalars.
    scalar_values: dict[str, ScalarValue] | None = None


class ExampleError(Exception):
    """An example folder is malformed: a missing or unreadable file, or inconsistent contents."""


def example_dir(root: Path, chapter: str, operator: str, variant: str, example: str) -> Path:
    """The folder of one example, ``<root>/<chapter>/<operator>/<variant>/<example>``."""
    return root / chapter / operator / variant / example


def scan_example_dirs(root: Path) -> set[tuple[str, str, str, str]]:
    """The ``(chapter, operator, variant, example)`` key of every folder with a ``script.vtl``.

    Only folders exactly four levels below ``root`` are examples: a ``script.vtl`` at any other
    depth is ignored.
    """
    keys: set[tuple[str, str, str, str]] = set()
    for script in root.glob(f"*/*/*/*/{_SCRIPT}"):
        if script.is_file():
            chapter, operator, variant, example = script.relative_to(root).parts[:4]
            keys.add((chapter, operator, variant, example))
    return keys


def read_csv_rows(path: Path) -> list[dict[str, str | None]]:
    """The rows of a CSV file, keyed by its header. An empty field is None, any other verbatim.

    A UTF-8 byte order mark is ignored. A repeated column name, or a row with more or fewer
    fields than the header, raises ``ExampleError``.
    """
    return _read_csv(path)[1]


def load_example(path: Path) -> ExampleFiles:
    """Load the example folder ``path``; raise ``ExampleError`` when it is malformed."""
    script_file = path / _SCRIPT
    if not script_file.is_file():
        raise ExampleError(f"{path}: missing {_SCRIPT}")
    if not (path / _EXPECTED).is_dir():
        raise ExampleError(f"{path}: missing {_EXPECTED}/ folder")
    script = _read_text(script_file)

    datasets = _load_expected_datasets(path)
    scalars = _load_expected_scalars(path / _EXPECTED / _SCALARS)
    error = _load_expected_error(path / _EXPECTED / _ERROR)
    if error is not None and (datasets or scalars):
        found = [f"{_EXPECTED}/{dataset.name}.json" for dataset in datasets]
        found += [f"{_EXPECTED}/{_SCALARS}"] if scalars else []
        raise ExampleError(
            f"{path / _EXPECTED / _ERROR}: an example expects an error or outputs, not both "
            f"(outputs found: {', '.join(found)})"
        )
    if error is None and not datasets and not scalars:
        raise ExampleError(
            f"{path}: no expected output (add {_EXPECTED}/<NAME>.json with "
            f"{_EXPECTED}/<NAME>.csv, {_EXPECTED}/{_SCALARS}, or {_EXPECTED}/{_ERROR} for an "
            "error)"
        )
    names = [dataset.name for dataset in datasets] + [scalar.name for scalar in scalars]
    for name in dict.fromkeys(names):
        if names.count(name) > 1:
            raise ExampleError(f"{path}: expected output '{name}' is defined twice")

    input_dir = path / _INPUT
    structures = _files(input_dir, ".json")
    # ``scalars.csv`` holds the values of the input scalars, not the data points of a dataset.
    datapoints = {
        csv_file.stem: csv_file
        for csv_file in _files(input_dir, ".csv")
        if csv_file.name != _SCALAR_VALUES
    }
    _check_datapoints(structures, datapoints)
    scalar_values = _load_input_scalars(path)
    return ExampleFiles(
        path=path,
        script=script,
        structures=structures,
        datapoints=datapoints,
        value_domains=_files(path / _VALUE_DOMAINS, ".json"),
        external_routines=_files(path / _EXTERNAL_ROUTINES, ".json"),
        expected_datasets=datasets,
        expected_scalars=scalars,
        expected_error=error,
        scalar_values=scalar_values,
    )


# --- Notes ---


def check_notes(text: str, example: Example) -> list[str]:
    """The problems of ``text``, the ``notes.rst`` of ``example``: none when it is well formed.

    The notes are a field list. Its fields are ``Tests``, ``Manual`` and ``Expected``, then any of
    ``Finding``, ``Erratum``, ``Adaptation`` and ``Team decision``, in this order and once each.
    ``Erratum`` is there when the example sets ``erratum`` and only then, ``Adaptation`` the same
    for ``adaptation`` and ``Team decision`` for ``decision``. A field is one paragraph: it starts
    with ``:<Name>:`` at the start of a line and goes on in indented lines, with no blank line
    inside it and nothing outside the fields. The text has 1200 characters at most.

    The field list is read with a regular expression, as the tool does not depend on docutils:
    ``check_notes_render.py`` parses the notes with docutils, in the venv of the docs.
    """
    problems: list[str] = []
    if len(text) > _MAX_NOTES:
        problems.append(f"{len(text)} characters: the notes have {_MAX_NOTES} at most")
    fields, layout = _note_fields(text)
    problems.extend(layout)
    names = [name for name, _ in fields]
    known = (*_REQUIRED_FIELDS, *_OPTIONAL_FIELDS)
    problems.extend(f"unknown field '{name}'" for name in names if name not in known)
    if not _in_layout([name for name in names if name in known]):
        problems.append(f"the fields are {', '.join(names) or 'none'}: {_LAYOUT}")
    for field, attribute in _FLAGGED_FIELDS:
        flagged = getattr(example, attribute) is not None
        if flagged and field not in names:
            problems.append(f"no field '{field}', but the example sets {attribute}")
        elif field in names and not flagged:
            problems.append(f"a field '{field}', but the example does not set {attribute}")
    return problems


def notes_problems(path: Path, example: Example) -> list[str]:
    """The problems of the notes of the example folder ``path``, ``example`` in the catalogue.

    The folder has a ``notes.rst`` that ``check_notes`` finds well formed, and no ``notes.md``:
    the notes of the first version of the map, which ``notes.rst`` replaces. Each problem names
    its file, or the folder for a ``notes.rst`` that is missing.
    """
    problems: list[str] = []
    old = path / _OLD_NOTES
    if old.exists():
        problems.append(f"{old}: the notes go in {_NOTES}, not in {_OLD_NOTES}")
    notes = path / _NOTES
    if not notes.is_file():
        problems.append(f"{path}: missing {_NOTES}")
        return problems
    try:
        text = _read_text(notes)
    except ExampleError as exc:
        return [*problems, str(exc)]
    problems.extend(f"{notes}: {problem}" for problem in check_notes(text, example))
    return problems


def _note_fields(text: str) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """The fields of the notes, each a name and the lines of its text, and the layout problems.

    A line that is not a field, nor the indented line of one, is outside the fields. An indented
    line after a blank line in a field would make a second paragraph of it.
    """
    fields: list[tuple[str, list[str]]] = []
    problems: list[str] = []
    gap = False  # a blank line since the last line of the field
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            gap = True
            continue
        found = _FIELD.match(f"{line}\n")
        if found is not None:
            first = line[found.end() :].strip()
            fields.append((found.group(1), [first] if first else []))
        elif not line[0].isspace() or not fields:
            problems.append(f"line {number} {_OUTSIDE}")
        else:
            if gap:
                name = fields[-1][0]
                problems.append(
                    f"line {number}: a blank line inside the field '{name}': a field is one "
                    "paragraph"
                )
            fields[-1][1].append(line.strip())
        gap = False
    problems.extend(f"the field '{name}' is empty" for name, lines in fields if not lines)
    return fields, problems


def _in_layout(names: list[str]) -> bool:
    """Whether ``names`` are the required fields, then optional ones in their order, each once."""
    head, rest = names[: len(_REQUIRED_FIELDS)], names[len(_REQUIRED_FIELDS) :]
    if tuple(head) != _REQUIRED_FIELDS or any(name not in _OPTIONAL_FIELDS for name in rest):
        return False
    positions = [_OPTIONAL_FIELDS.index(name) for name in rest]
    return all(before < after for before, after in zip(positions, positions[1:]))


# --- Files ---


def _files(folder: Path, suffix: str) -> list[Path]:
    """The files of ``folder`` that end with ``suffix``, sorted by name; none if it is absent."""
    found = (file for file in folder.glob(f"*{suffix}") if file.is_file())
    return sorted(found, key=lambda file: file.name)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExampleError(f"{path}: not valid UTF-8 ({exc})") from exc


def _read_json(path: Path, **options: Any) -> Any:
    try:
        return json.loads(_read_text(path), **options)
    except json.JSONDecodeError as exc:
        raise ExampleError(f"{path}: invalid JSON ({exc})") from exc


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str | None]]]:
    """The header and the rows of a CSV file."""
    header: list[str] = []
    rows: list[dict[str, str | None]] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            for name in header:
                if header.count(name) > 1:
                    raise ExampleError(f"{path}: duplicate column '{name}'")
            for row in reader:
                # DictReader keys the surplus fields by None and fills the missing ones with None.
                if None in row:
                    raise ExampleError(
                        f"{path}: line {reader.line_num} has more fields than the header"
                    )
                if None in row.values():
                    raise ExampleError(
                        f"{path}: line {reader.line_num} has fewer fields than the header"
                    )
                rows.append({name: value or None for name, value in row.items()})
    except UnicodeDecodeError as exc:
        raise ExampleError(f"{path}: not valid UTF-8 ({exc})") from exc
    except csv.Error as exc:
        raise ExampleError(f"{path}: invalid CSV ({exc})") from exc
    return header, rows


# --- Expected outputs ---


def _expect(value: Any, kind: type[_T], file: Path, what: str) -> _T:
    """``value`` when it is a ``kind``, else an ``ExampleError`` about ``what`` of ``file``."""
    if not isinstance(value, kind):
        raise ExampleError(f"{file}: {what} must be {_KINDS[kind]}")
    return value


def _load_expected_datasets(path: Path) -> list[ExpectedDataset]:
    """Every ``expected/<NAME>.json`` with its ``expected/<NAME>.csv``."""
    folder = path / _EXPECTED
    # The other JSON files of ``expected/`` are not the structure of a dataset.
    others = (_SCALARS, _ERROR)
    structures = {file.stem: file for file in _files(folder, ".json") if file.name not in others}
    data = {file.stem: file for file in _files(folder, ".csv")}
    unpaired = sorted(structures.keys() ^ data.keys())
    if unpaired:
        stem = unpaired[0]
        have, need = ("json", "csv") if stem in structures else ("csv", "json")
        raise ExampleError(
            f"{path}: missing {_EXPECTED}/{stem}.{need} (it goes with {_EXPECTED}/{stem}.{have})"
        )
    return [_load_expected_dataset(file, data[stem]) for stem, file in structures.items()]


def _load_expected_dataset(json_file: Path, csv_file: Path) -> ExpectedDataset:
    document = _expect(_read_json(json_file), dict, json_file, "the document")
    entries = _expect(document.get("datasets"), list, json_file, "'datasets'")
    if len(entries) != 1:
        raise ExampleError(f"{json_file}: must describe exactly one dataset, found {len(entries)}")
    entry = _expect(entries[0], dict, json_file, "the dataset")
    name = _expect(entry.get("name"), str, json_file, "the dataset name")
    if name != json_file.stem:
        raise ExampleError(f"{json_file}: the dataset is named '{name}', not '{json_file.stem}'")

    components: dict[str, ExpectedComponent] = {}
    for raw in _expect(entry.get("DataStructure"), list, json_file, "'DataStructure'"):
        component_name, component = _parse_component(raw, json_file)
        if component_name in components:
            raise ExampleError(f"{json_file}: duplicate component '{component_name}'")
        components[component_name] = component

    header, rows = _read_csv(csv_file)
    _check_columns(csv_file, header, json_file, components)
    return ExpectedDataset(name=name, components=components, rows=rows)


def _check_columns(
    csv_file: Path, header: list[str], json_file: Path, declared: Collection[str]
) -> None:
    """The CSV columns are the names of the structure: a mismatch is a typo in one file.

    The names are the components of a dataset, or the input scalars.
    """
    missing = [column for column in declared if column not in header]
    unknown = [column for column in header if column not in declared]
    problems = []
    if missing:
        problems.append(f"missing {', '.join(missing)}")
    if unknown:
        problems.append(f"not in the structure: {', '.join(unknown)}")
    if problems:
        raise ExampleError(
            f"{csv_file}: columns do not match {json_file.name} ({'; '.join(problems)})"
        )


def _parse_component(raw: Any, file: Path) -> tuple[str, ExpectedComponent]:
    item = _expect(raw, dict, file, "a component")
    name = _expect(item.get("name"), str, file, "a component name")
    role = _expect(item.get("role"), str, file, f"the role of {name}")
    raw_nullable = item.get("nullable")
    nullable = (
        None if raw_nullable is None else _expect(raw_nullable, bool, file, f"nullable of {name}")
    )
    return name, ExpectedComponent(role, _parse_types(item.get("type"), file, name), nullable)


def _parse_types(raw: Any, file: Path, component: str) -> tuple[str, ...]:
    """The ``type`` of a component: a string, or a non-empty list of strings."""
    types = raw if isinstance(raw, list) else [raw]
    if not types or not all(isinstance(type_, str) for type_ in types):
        raise ExampleError(
            f"{file}: the type of {component} must be a string or a non-empty list of strings"
        )
    return tuple(types)


def _load_expected_scalars(file: Path) -> list[ExpectedScalar]:
    """The scalars of ``expected/scalars.json``; none when the file is absent."""
    if not file.is_file():
        return []
    # Numbers are read as written (``parse_int``/``parse_float`` get their text), so that the
    # printed precision, ``10.50`` or ``1E-7``, reaches the comparison.
    document = _expect(_read_json(file, parse_int=str, parse_float=str), dict, file, "the document")
    scalars: list[ExpectedScalar] = []
    for raw in _expect(document.get("scalars"), list, file, "'scalars'"):
        item = _expect(raw, dict, file, "a scalar")
        name = _expect(item.get("name"), str, file, "a scalar name")
        type_ = _expect(item.get("type"), str, file, f"the type of {name}")
        if "value" not in item:
            raise ExampleError(f"{file}: scalar '{name}' has no value (null is written null)")
        scalars.append(ExpectedScalar(name, type_, _scalar_value(item["value"], file, name)))
    return scalars


def _load_expected_error(file: Path) -> str | None:
    """The code of ``expected/error.json``, ``{"code": "2-1-15-6"}``; None without the file."""
    if not file.is_file():
        return None
    document = _expect(_read_json(file), dict, file, "the document")
    code = _expect(document.get("code"), str, file, "the code")
    if ERROR_CODE.fullmatch(code) is None:
        raise ExampleError(
            f"{file}: {code!r} is not the code of a vtlengine error "
            "(digits in groups joined by '-', such as 2-1-15-6)"
        )
    return code


def _scalar_value(raw: Any, file: Path, name: str) -> str | None:
    if raw is None or isinstance(raw, str):
        return raw
    if isinstance(raw, bool):
        return "true" if raw else "false"
    raise ExampleError(f"{file}: the value of {name} must be a string, a number, a boolean or null")


# --- Input scalars ---


def _load_input_scalars(path: Path) -> dict[str, ScalarValue] | None:
    """The values of ``input/scalars.csv``, typed by the scalars of ``input/scalars.json``.

    The two files go together: an example has both or neither. The CSV has a column for each
    declared scalar and no other, and exactly one data row. An example without them has no input
    scalar values: None.
    """
    structure = path / _INPUT / _SCALARS
    values = path / _INPUT / _SCALAR_VALUES
    if structure.is_file() != values.is_file():
        have, need = (
            (_SCALARS, _SCALAR_VALUES) if structure.is_file() else (_SCALAR_VALUES, _SCALARS)
        )
        raise ExampleError(f"{path}: missing {_INPUT}/{need} (it goes with {_INPUT}/{have})")
    if not structure.is_file():
        return None
    types = _declared_scalars(structure)
    header, rows = _read_csv(values)
    _check_columns(values, header, structure, types)
    if len(rows) != 1:
        raise ExampleError(f"{values}: must have exactly one data row, found {len(rows)}")
    return {name: _input_value(rows[0][name], type_, values, name) for name, type_ in types.items()}


def _declared_scalars(file: Path) -> dict[str, str]:
    """The scalars of ``input/scalars.json`` with their VTL types, in the order of the file."""
    document = _expect(_read_json(file), dict, file, "the document")
    types: dict[str, str] = {}
    for raw in _expect(document.get("scalars"), list, file, "'scalars'"):
        item = _expect(raw, dict, file, "a scalar")
        name = _expect(item.get("name"), str, file, "a scalar name")
        type_ = _expect(item.get("type"), str, file, f"the type of {name}")
        if type_ not in _SCALAR_TYPES:
            raise ExampleError(
                f"{file}: the type of {name} must be one of {', '.join(_SCALAR_TYPES)}, "
                f"not {type_!r}"
            )
        if name in types:
            raise ExampleError(f"{file}: duplicate scalar '{name}'")
        types[name] = type_
    return types


def _input_value(text: str | None, type_: str, file: Path, name: str) -> ScalarValue:
    """The value of the input scalar ``name`` of ``type_``, from its cell ``text`` of ``file``.

    An empty cell is None. An Integer, a Number and a Boolean are the Python types that ``run``
    takes for them. Any other type is its text, as the file writes it, which the engine checks.
    """
    if text is None or type_ not in _READERS:
        return text
    read, wanted = _READERS[type_]
    try:
        return read(text)
    except ValueError:
        raise ExampleError(f"{file}: the value of {name} must be {wanted}, not {text!r}") from None


def _read_integer(text: str) -> int:
    if _INTEGER.fullmatch(text) is None:
        raise ValueError(text)
    return int(text)


def _read_number(text: str) -> float:
    value = float(text) if _NUMBER.fullmatch(text) is not None else None
    if value is None or not math.isfinite(value):  # ``1e999`` is no number once it is read
        raise ValueError(text)
    return value


def _read_boolean(text: str) -> bool:
    if text.lower() not in ("true", "false"):
        raise ValueError(text)
    return text.lower() == "true"


# How the values of the types that are not text are read, and what the file must hold for each.
_READERS: Final[dict[str, tuple[Callable[[str], ScalarValue], str]]] = {
    "Integer": (_read_integer, "an Integer"),
    "Number": (_read_number, "a Number"),
    "Boolean": (_read_boolean, "true or false"),
}


# --- Input datasets ---


def _check_datapoints(structures: list[Path], datapoints: dict[str, Path]) -> None:
    """Every ``input/<NAME>.csv`` is the data of a dataset that some structure file declares.

    ``run`` refuses any other CSV (``Not found dataset NAME in datastructures``) on both engines,
    so a typo in a file name would look like an engine defect instead of an authoring error.
    """
    declared = _declared_datasets(structures)
    for name, csv_file in datapoints.items():
        if name not in declared:
            known = ", ".join(sorted(declared)) or "none"
            raise ExampleError(
                f"{csv_file}: no structure declares the dataset '{name}' "
                f"(declared in {_INPUT}/*.json: {known})"
            )


def _declared_datasets(structures: list[Path]) -> set[str]:
    """The names of the ``datasets`` of the structure files, as the engine reads them."""
    names: set[str] = set()
    for file in structures:
        document = _expect(_read_json(file), dict, file, "the document")
        for entry in _expect(document.get("datasets", []), list, file, "'datasets'"):
            dataset = _expect(entry, dict, file, "a dataset")
            names.add(_expect(dataset.get("name"), str, file, "the dataset name"))
    return names
