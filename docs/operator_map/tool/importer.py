"""Import of the VTL 2.2 Reference Manual's examples into the example folders of the map.

The manual keeps the examples of a page in ``operators/<Chapter>/<Page>/examples/``:

* ``ex_<n>.vtl``, ``ex_<n>.json`` and ``ex_<n>.csv``: the script of an example and its result,
  which is a dataset (structure and data points) or a scalar (type and value);
* ``ds_<n>.json`` and ``ds_<n>.csv``: the structure and the data points of an input dataset.
  The examples of a page share its inputs, and each script uses some of them.

``build_index_source`` lists what the manual holds, for the coverage checks of the catalogue.
``import_example`` turns one manual example into an example folder (see ``tool.examples``).
What it cannot settle by itself, such as a ruleset that only the text of the page defines, comes
back as warnings for whoever adds the example to the map.

Two choices are worth knowing. The manual does not say whether the components of a result are
nullable, so the expected structures leave ``nullable`` out and the comparison does not check it.
And a ``ds_<n>.json`` that declares another dataset than its file name suggests (the manual has
a few such slips) is imported under its file name's dataset, with a warning.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from tool.examples import example_dir
from tool.model import DL_PAGES, Chapter

_OPERATORS: Final = "operators"
_EXAMPLES: Final = "examples"
_SCRIPT: Final = "script.vtl"
_INPUT: Final = "input"
_EXPECTED: Final = "expected"
_SCALARS_FILE: Final = "scalars.json"
_IDENTIFIER: Final = "Identifier"
_STRING: Final = "String"

# The manual's names of the types, in lowercase, to the names of vtlengine's structure files.
_TYPE_NAMES: Final = {
    "integer": "Integer",
    "number": "Number",
    "string": _STRING,
    "boolean": "Boolean",
    "date": "Date",
    "time": "Time",
    "time_period": "Time_Period",
    "timeperiod": "Time_Period",
    "duration": "Duration",
}
_ROLES: Final = frozenset(
    {_IDENTIFIER, "Measure", "Attribute", "ViralAttribute", "Viral Attribute"}
)

_COMMIT: Final = re.compile(r"[0-9a-f]{40}")
_EXAMPLE_SCRIPT: Final = re.compile(r"ex_\d+\.vtl")

# What the scripts are searched for, once ``_code`` has blanked their comments and strings.
_LITERAL_OR_COMMENT: Final = re.compile(r'"[^"]*"|/\*.*?\*/|//[^\n]*', re.DOTALL)
_ASSIGNED: Final = re.compile(r"(?:^|;)\s*(\w+)\s*(?::=|<-)", re.MULTILINE)
_DATASET_NAME: Final = re.compile(r"\bDS_\w+")
_RULESET_CALL: Final = re.compile(r"\b(?:hierarchy|check_hierarchy|check_datapoint)\s*\(")
_RULESET_DEFINITION: Final = re.compile(r"\bdefine\s+(?:datapoint|hierarchical)\s+ruleset\s+(\w+)")
_VALUE_DOMAIN_USE: Final = re.compile(r"\b(?:in|not_in)\s+([A-Za-z_]\w*)")
_NAME: Final = re.compile(r"\s*([A-Za-z_]\w*)")


class ManualImportError(Exception):
    """The manual, or what is asked of it, cannot be imported: a file is missing or malformed."""


@dataclass(frozen=True)
class ChapterImport:
    """What importing the manual examples of a chapter did."""

    imported: int  # the number of example folders written
    skipped: list[str]  # the example folders left alone because they exist
    warnings: list[str]  # each led by the example folder it is about


# --- Index of the manual ---


def list_manual(spec_root: Path) -> tuple[list[str], list[str]]:
    """The pages of the manual, and its examples; both sorted.

    ``spec_root`` is ``v2.2/docs/reference_manual`` of the manual's repository. A page is a folder
    ``operators/<Chapter>/<Page>`` with an ``index.rst``: it is named ``<Chapter>/<Page>``. An
    example is an ``ex_<n>.vtl`` of the ``examples`` folder of a page: it is named
    ``<Chapter>/<Page>/ex_<n>``.
    """
    operators = spec_root / _OPERATORS
    if not operators.is_dir():
        raise ManualImportError(f"{spec_root} has no '{_OPERATORS}' folder: it is not the manual")
    pages = sorted(
        index.parent.relative_to(operators).as_posix() for index in operators.glob("*/*/index.rst")
    )
    examples = sorted(
        f"{page}/{script.stem}"
        for page in pages
        for script in (operators / page / _EXAMPLES).glob("ex_*.vtl")
        if _EXAMPLE_SCRIPT.fullmatch(script.name)
    )
    return pages, examples


def build_index_source(spec_root: Path, commit: str) -> str:
    """The source of ``catalogue/manual_index.py`` for the manual at ``commit``.

    It holds ``MANUAL_COMMIT``, ``MANUAL_PAGES`` (the pages of the manual and ``DL_PAGES``) and
    ``MANUAL_EXAMPLES``, the last two as sorted tuples. Nothing in it varies from run to run.
    """
    if _COMMIT.fullmatch(commit) is None:
        raise ManualImportError(f"'{commit}' is not a full commit hash (40 lowercase hex digits)")
    pages, examples = list_manual(spec_root)
    lines = [
        '"""The pages and the examples of the VTL 2.2 Reference Manual.',
        "",
        "Generated by ``import_manual.py --index``, from the manual at ``MANUAL_COMMIT``.",
        "Do not edit it.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Final",
        "",
        f"MANUAL_COMMIT: Final = {_literal(commit)}",
        "",
        "# The ``<chapter>/<page>`` of the pages of the manual, and the pages of the VTL-DL.",
        *_tuple_lines("MANUAL_PAGES", sorted({*pages, *DL_PAGES})),
        "",
        "# The ``<chapter>/<page>/ex_<n>`` of the examples of the manual.",
        *_tuple_lines("MANUAL_EXAMPLES", examples),
    ]
    return "\n".join(lines) + "\n"


def _literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _tuple_lines(name: str, items: Sequence[str]) -> list[str]:
    head = f"{name}: Final[tuple[str, ...]] = "
    if not items:
        return [f"{head}()"]
    return [f"{head}(", *(f"    {_literal(item)}," for item in items), ")"]


# --- Conversion of the structures ---


def convert_structure(manual_json: dict[str, Any]) -> dict[str, Any]:
    """The structure of a dataset or a scalar of the manual, in vtlengine's format.

    A dataset ``{"name", "components": [{"name", "role", "data_type"}]}`` becomes
    ``{"datasets": [{"name", "DataStructure": [{"name", "type", "role", "nullable"}]}]}``: the
    types take the names of vtlengine (``TimePeriod`` is ``Time_Period``) and a component is
    nullable unless it is an identifier. A scalar, a JSON with a ``type`` and no components,
    becomes ``{"scalars": [{"name", "type"}]}``.
    """
    name = _text(manual_json.get("name"), "the name")
    if "components" in manual_json:
        raw = manual_json["components"]
        if not isinstance(raw, list):
            raise ManualImportError(f"the components of {name} must be a list")
        components = [_component(item, name) for item in raw]
        return {"datasets": [{"name": name, "DataStructure": components}]}
    if "type" in manual_json:
        return {"scalars": [{"name": name, "type": _canonical_type(manual_json["type"], name)}]}
    raise ManualImportError(f"{name} has neither components nor a type")


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManualImportError(f"{what} must be a text")
    return value


def _component(raw: Any, dataset: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ManualImportError(f"a component of {dataset} must be an object")
    name = _text(raw.get("name"), f"the name of a component of {dataset}")
    role = _text(raw.get("role"), f"the role of {name}")
    if role not in _ROLES:
        raise ManualImportError(f"unknown role '{role}' of {name}")
    return {
        "name": name,
        "type": _canonical_type(raw.get("data_type"), name),
        "role": role,
        "nullable": role != _IDENTIFIER,
    }


def _canonical_type(raw: Any, owner: str) -> str:
    canonical = _TYPE_NAMES.get(raw.lower()) if isinstance(raw, str) else None
    if canonical is None:
        raise ManualImportError(f"unknown type {raw!r} of {owner}")
    return canonical


def _types(dataset: dict[str, Any]) -> dict[str, str]:
    """The type of every component of a converted dataset."""
    return {component["name"]: component["type"] for component in dataset["DataStructure"]}


# --- Files ---


def _read_text(path: Path) -> str:
    """The text of a file, without a byte order mark and with ``\\n`` line ends."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ManualImportError(f"{path}: cannot be read ({exc.strerror or exc})") from exc
    except UnicodeDecodeError as exc:
        raise ManualImportError(f"{path}: not valid UTF-8 ({exc})") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise ManualImportError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(document, dict):
        raise ManualImportError(f"{path}: the document must be an object")
    return document


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """The header and the data rows of a CSV file."""
    rows: list[list[str]] = []
    reader = csv.reader(io.StringIO(_read_text(path), newline=""))
    try:
        header = next(reader, None)
        if header is None:
            raise ManualImportError(f"{path}: the file is empty")
        for row in reader:
            if not row and len(header) == 1:
                row = [""]  # the only field of the line is empty
            if len(row) != len(header):
                raise ManualImportError(
                    f"{path}: line {reader.line_num} has {len(row)} fields, "
                    f"the header has {len(header)}"
                )
            rows.append(row)
    except csv.Error as exc:
        raise ManualImportError(f"{path}: invalid CSV ({exc})") from exc
    return header, rows


def _read_data(csv_file: Path, types: dict[str, str]) -> str:
    """The data points of a dataset as the example folder keeps them.

    The cells of the columns that are not Strings lose the whitespace around them, which the
    manual leaves in some numbers (``2.1     ``). The String cells are kept as they are.
    """
    header, rows = _read_csv(csv_file)
    missing = [name for name in types if name not in header]
    unknown = [name for name in header if name not in types]
    if missing or unknown:
        raise ManualImportError(
            f"{csv_file}: columns do not match the structure "
            f"(missing: {', '.join(missing) or 'none'}; "
            f"not in the structure: {', '.join(unknown) or 'none'})"
        )
    verbatim = [types[name] == _STRING for name in header]
    cleaned = [[c if keep else c.strip() for c, keep in zip(row, verbatim)] for row in rows]
    return "".join(",".join(_quote(cell) for cell in row) + "\n" for row in [header, *cleaned])


def _quote(cell: str) -> str:
    """A CSV field, quoted when it holds a delimiter or a quote, or begins or ends with a space."""
    if cell and (cell != cell.strip() or any(char in cell for char in ',"\r\n')):
        return '"' + cell.replace('"', '""') + '"'
    return cell


def _json_text(document: dict[str, Any]) -> str:
    """JSON with one line per component or scalar, which keeps a structure readable."""
    return _render(document, 0) + "\n"


def _render(value: Any, level: int) -> str:
    if isinstance(value, dict) and any(isinstance(item, (dict, list)) for item in value.values()):
        pad = "  " * (level + 1)
        items = [f"{pad}{_literal(key)}: {_render(item, level + 1)}" for key, item in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"
    if isinstance(value, list) and value:
        pad = "  " * (level + 1)
        items = [f"{pad}{_render(item, level + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + "\n" + "  " * level + "]"
    return json.dumps(value, ensure_ascii=False)


# --- Import of one example ---


@dataclass
class _Plan:
    """What importing an example writes, and what it warns about."""

    files: dict[str, str] = field(default_factory=dict)  # path in the example folder -> text
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ManualInput:
    """An input dataset of a page: a ``ds_<n>.json`` of its examples folder."""

    json_file: Path
    declared: str  # the name the file gives the dataset
    name: str  # the name the scripts use: the declared one, unless the file name says another


def import_example(
    spec_root: Path, page: str, ref: str, dest: Path, force: bool = False
) -> list[str]:
    """Import the manual example ``<page>/<ref>`` into the example folder ``dest``.

    ``page`` is ``<Chapter>/<Page>`` and ``ref`` is ``ex_<n>``. The folder gets ``script.vtl``,
    the inputs the script uses in ``input/`` and the result in ``expected/``. A destination that
    already has any of them is left as it is unless ``force`` is set, so that the corrections
    made by hand survive a second import. Then the three are replaced, and the rest of the
    folder (``notes.md``, ``value_domains/``...) is kept. The result is the list of warnings, or
    the reason an existing destination was skipped. Nothing is written when the example cannot
    be read.
    """
    return _import_example(spec_root, page, ref, dest, force)[1]


def _import_example(
    spec_root: Path, page: str, ref: str, dest: Path, force: bool
) -> tuple[bool, list[str]]:
    """``import_example``, and whether it wrote the folder."""
    if dest.exists() and not dest.is_dir():
        raise ManualImportError(f"{dest} exists and is not a folder")
    owned = [dest / name for name in (_SCRIPT, _INPUT, _EXPECTED)]
    if not force and any(path.exists() for path in owned):
        return False, ["the example folder exists: skipped (--force replaces it)"]
    plan = _plan_example(spec_root, page, ref)
    for path in owned:
        _remove(path)
    for name, text in plan.files.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return True, plan.warnings


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _plan_example(spec_root: Path, page: str, ref: str) -> _Plan:
    folder = spec_root / _OPERATORS / page / _EXAMPLES
    script = _read_text(folder / f"{ref}.vtl")
    code = _code(script)
    assigned = set(_ASSIGNED.findall(code))
    plan = _Plan(files={_SCRIPT: script.rstrip() + "\n"})
    output = _plan_expected(plan, folder, ref)
    imported = _plan_inputs(plan, folder, code, assigned)
    _warn_on_references(plan, code, assigned, imported, output)
    return plan


def _code(script: str) -> str:
    """The script without its comments and without what its string constants say."""
    return _LITERAL_OR_COMMENT.sub(lambda m: '""' if m.group().startswith('"') else " ", script)


def _convert(json_file: Path, document: dict[str, Any]) -> dict[str, Any]:
    try:
        return convert_structure(document)
    except ManualImportError as exc:
        raise ManualImportError(f"{json_file}: {exc}") from exc


def _convert_dataset(json_file: Path, document: dict[str, Any]) -> dict[str, Any]:
    structure = _convert(json_file, document)
    if "datasets" not in structure:
        raise ManualImportError(f"{json_file}: a dataset was expected, not a scalar")
    return structure


def _plan_expected(plan: _Plan, folder: Path, ref: str) -> str:
    """Add the expected output of the example to ``plan``; return its name."""
    json_file = folder / f"{ref}.json"
    csv_file = folder / f"{ref}.csv"
    structure = _convert(json_file, _read_json(json_file))
    if "scalars" in structure:
        scalar = structure["scalars"][0]
        scalar["value"] = _scalar_value(csv_file, scalar["type"])
        plan.files[f"{_EXPECTED}/{_SCALARS_FILE}"] = _json_text(structure)
        return str(scalar["name"])
    dataset = structure["datasets"][0]
    for component in dataset["DataStructure"]:
        del component["nullable"]  # the manual does not say
    name = str(dataset["name"])
    plan.files[f"{_EXPECTED}/{name}.json"] = _json_text(structure)
    plan.files[f"{_EXPECTED}/{name}.csv"] = _read_data(csv_file, _types(dataset))
    return name


def _scalar_value(csv_file: Path, type_: str) -> str | None:
    """The value of a scalar result: the first data cell of its file, None when it is empty."""
    _, rows = _read_csv(csv_file)
    if not rows:
        raise ManualImportError(f"{csv_file}: the file has no value")
    cell = rows[0][0]
    return (cell if type_ == _STRING else cell.strip()) or None


def _manual_inputs(folder: Path) -> list[_ManualInput]:
    """The ``ds_<n>.json`` (or ``DS_<n>.json``) files of an examples folder."""
    inputs: list[_ManualInput] = []
    for json_file in sorted(folder.glob("*.json")):
        if not json_file.stem.lower().startswith("ds_"):
            continue
        declared = _text(_read_json(json_file).get("name"), f"{json_file}: the dataset name")
        agrees = declared.lower() == json_file.stem.lower()
        inputs.append(
            _ManualInput(json_file, declared, declared if agrees else json_file.stem.upper())
        )
    return inputs


def _plan_inputs(plan: _Plan, folder: Path, code: str, assigned: set[str]) -> set[str]:
    """Add the inputs that the script uses to ``plan``; return their names.

    An input is used when its name occurs as a whole word in the script and the script does not
    assign it.
    """
    imported: set[str] = set()
    for item in _manual_inputs(folder):
        if item.name in assigned or re.search(rf"\b{re.escape(item.name)}\b", code) is None:
            continue
        if item.name in imported:
            raise ManualImportError(f"{folder}: two files give the dataset {item.name}")
        imported.add(item.name)
        if item.declared != item.name:
            plan.warnings.append(
                f"{item.json_file.name} declares the dataset {item.declared}: "
                f"imported as {item.name}, the dataset its file name says"
            )
        document = {**_read_json(item.json_file), "name": item.name}
        structure = _convert_dataset(item.json_file, document)
        plan.files[f"{_INPUT}/{item.name}.json"] = _json_text(structure)
        csv_file = item.json_file.with_suffix(".csv")
        if csv_file.is_file():
            data = _read_data(csv_file, _types(structure["datasets"][0]))
            plan.files[f"{_INPUT}/{item.name}.csv"] = data
        else:
            plan.warnings.append(f"{csv_file.name} is missing: {item.name} is imported empty")
    return imported


def _warn_on_references(
    plan: _Plan, code: str, assigned: set[str], imported: set[str], output: str
) -> None:
    """Warn about what the script needs and the import cannot give it."""
    for name in sorted(set(_DATASET_NAME.findall(code)) - assigned - imported):
        plan.warnings.append(f"{name} is used by the script but the page has no input file for it")
    if output not in assigned:
        assigns = f"; it assigns {', '.join(sorted(assigned))}" if assigned else ""
        plan.warnings.append(f"the expected output {output} is not assigned by the script{assigns}")
    defined = set(_RULESET_DEFINITION.findall(code))
    for name in dict.fromkeys(_ruleset_references(code)):
        if name not in defined:
            plan.warnings.append(f"ruleset {name} is used but not defined in the script")
    for name in dict.fromkeys(_VALUE_DOMAIN_USE.findall(code)):
        plan.warnings.append(f"value domain {name} is used but not defined in the script")


def _ruleset_references(code: str) -> list[str]:
    """The rulesets that ``hierarchy``, ``check_hierarchy`` and ``check_datapoint`` name.

    The ruleset is the identifier after the first comma of the call that is not nested in the
    first operand, which can be an expression with commas of its own.
    """
    names: list[str] = []
    for call in _RULESET_CALL.finditer(code):
        comma = _first_comma(code, call.end())
        found = _NAME.match(code, comma + 1) if comma is not None else None
        if found is not None:
            names.append(found.group(1))
    return names


def _first_comma(code: str, start: int) -> int | None:
    """The position of the first comma at the level of ``start``, None if the call ends first."""
    depth = 0
    for position in range(start, len(code)):
        char = code[position]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                return None
            depth -= 1
        elif char == "," and depth == 0:
            return position
    return None


# --- Import of a chapter ---


def import_chapter(
    chapter: Chapter, spec_root: Path, examples_root: Path, force: bool = False
) -> ChapterImport:
    """Import every manual example of ``chapter`` into its folder under ``examples_root``.

    The folder of an example is the one ``tool.examples.example_dir`` gives, and the folders are
    named relative to ``examples_root`` in the result. The authored examples are not touched.
    """
    imported = 0
    skipped: list[str] = []
    warnings: list[str] = []
    for operator in chapter.operators:
        for variant in operator.variants:
            for example in variant.examples:
                if example.source != "manual" or example.manual_ref is None:
                    continue
                dest = example_dir(examples_root, chapter.id, operator.id, variant.id, example.name)
                label = dest.relative_to(examples_root).as_posix()
                try:
                    written, found = _import_example(
                        spec_root, operator.page, example.manual_ref, dest, force
                    )
                except ManualImportError as exc:
                    raise ManualImportError(f"{label}: {exc}") from exc
                if written:
                    imported += 1
                    warnings.extend(f"{label}: {message}" for message in found)
                else:
                    skipped.append(label)
    return ChapterImport(imported, skipped, warnings)
