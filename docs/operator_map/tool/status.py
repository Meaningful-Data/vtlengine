"""Verdicts of the engines on an example, the status they make, and the roll-up of statuses.

``verdict`` says what one engine gave for an example: ``match`` when it gave the expected
outputs, ``mismatch`` when it ran but gave something else, ``error`` when it did not run.
``example_result`` puts the verdicts of the two engines together into the status of the example:

* ``diverge``: the engines do not agree. Two engines agree when both fail, whatever their
  messages, or when they have the same verdict and ``outputs_equal`` finds their outputs equal;
* ``ok``: both engines match;
* ``missing``: both engines fail;
* ``partial``: the engines agree and both give a wrong answer.

Only the outputs that the expected files of the example name are looked at, on both sides. A
script may return more, such as intermediate datasets: those are ignored. An expected output that
a run does not have is a mismatch. The examples that are checked by property (``random`` and
``current_date``) are judged by their values, not by expected values; the engines still have to
give the same values, so ``random`` diverges while the engines use different generators.

``rollup`` carries the statuses up from examples to variants, operators and chapters, and
``build_index`` does it for a whole catalogue.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from tool.compare import (
    ActualScalar,
    Output,
    compare_dataset,
    compare_scalar,
    normalize_type,
    outputs_equal,
    values_equal,
)
from tool.examples import ExampleFiles
from tool.model import Catalogue, Example, Status, Variant
from tool.runner import RunOutput

EMOJI: Final[dict[str, str]] = {"ok": "🟢", "partial": "🟡", "missing": "🔴", "diverge": "🔵"}

Outcome = Literal["match", "mismatch", "error"]

_STATUSES: Final[tuple[Status, ...]] = ("ok", "partial", "missing", "diverge")
_TYPES: Final = "types"  # the chapter, in the index, of the type hubs
_IDENTIFIER: Final = "Identifier"

# A decimal number in ASCII digits. ``float`` alone would also take ``nan``, ``inf`` and ``1_0``.
_NUMBER: Final = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)


@dataclass(frozen=True)
class EngineVerdict:
    """What an engine gave for an example; ``detail`` is empty for a ``match``."""

    outcome: Outcome
    detail: str


@dataclass(frozen=True)
class ExampleResult:
    """The verdicts of both engines on an example, whether they agree, and the status."""

    pandas: EngineVerdict
    duckdb: EngineVerdict
    engines_agree: bool
    status: Status


@dataclass(frozen=True)
class StatusIndex:
    """The statuses of a catalogue at every level, by key; the keys are in catalogue order.

    The type hubs are keyed like operators under ``types``: ``types/<type>/<variant>/<example>``
    for an example, ``types/<type>/<variant>`` for a variant, ``types/<type>`` in ``operators``
    and ``types`` in ``chapters``. A type with no variants has no key.
    """

    examples: dict[str, ExampleResult]  # "chapter/op/variant/example", "types/<type>/..."
    variants: dict[str, Status]  # "chapter/op/variant"
    operators: dict[str, Status]  # "chapter/op" and "types/<type>"
    chapters: dict[str, Status]  # "chapter" and "types"


# --- Verdicts ---


def verdict(example: Example, files: ExampleFiles, out: RunOutput, today: str) -> EngineVerdict:
    """What the run ``out`` of ``example`` on an engine gave: a match, a mismatch or an error.

    A run that failed is an ``error``, whose detail is ``<type> <code>: <message>`` with the
    parts that are missing left out. Otherwise every expected output must be in the run, by its
    name (an output that is not named in the expected files is ignored), or the verdict is a
    ``mismatch`` with the detail ``missing output <name>``. Then, by the ``check`` of the example:

    * ``exact``: every expected dataset and scalar must compare as equal, or the verdict is a
      ``mismatch`` and the detail is the first problem found;
    * ``random``: the run must be ``stable`` (the same on every run) and have at least one value,
      and every non-null value of a component that is not an identifier, and of a scalar, must
      be a number from 0 up to, but not including, 1;
    * ``current_date``: there must be at least one Date value, and every non-null Date value,
      of a dataset or a scalar, must be ``today``.
    """
    if not out.ok:
        return EngineVerdict("error", _error_detail(out))
    problem = _missing_output(files, out)
    if problem is None:
        problem = _check_problem(example, files, out, today)
    if problem is None:
        return EngineVerdict("match", "")
    return EngineVerdict("mismatch", problem)


def example_result(
    example: Example, files: ExampleFiles, pandas: RunOutput, duckdb: RunOutput, today: str
) -> ExampleResult:
    """The verdicts of ``example`` on both engines, whether the engines agree, and its status.

    The engines agree when both fail, and not when their verdicts differ. When both run to the
    same verdict they agree if the outputs that the example expects are equal on both
    (``outputs_equal``). The status is ``diverge`` when they do not agree, ``ok`` when both
    match, ``missing`` when both fail, and ``partial`` when both give the same wrong answer.
    """
    left = verdict(example, files, pandas, today)
    right = verdict(example, files, duckdb, today)
    agree = _engines_agree(files, left, right, pandas, duckdb)
    return ExampleResult(left, right, agree, _status(left, right, agree))


def _error_detail(out: RunOutput) -> str:
    """``<type> <code>: <message>`` of a failed run, with what is missing left out."""
    head = " ".join(part for part in (out.error_type, out.error_code) if part)
    if head and out.error_message:
        return f"{head}: {out.error_message}"
    return head or out.error_message or ""


def _missing_output(files: ExampleFiles, out: RunOutput) -> str | None:
    """The problem of the first expected output that the run does not have, or None.

    An expected dataset is looked for among the datasets of the run, and an expected scalar
    among its scalars.
    """
    datasets = {dataset.name for dataset in out.datasets}
    scalars = {scalar.name for scalar in out.scalars}
    absent = [wanted.name for wanted in files.expected_datasets if wanted.name not in datasets]
    absent += [wanted.name for wanted in files.expected_scalars if wanted.name not in scalars]
    return f"missing output {absent[0]}" if absent else None


def _check_problem(example: Example, files: ExampleFiles, out: RunOutput, today: str) -> str | None:
    """The problem that the ``check`` of the example finds in the run, or None."""
    if example.check == "exact":
        return _exact_problem(files, out)
    if example.check == "random":
        return _random_problem(files, out)
    if example.check == "current_date":
        return _current_date_problem(files, out, today)
    raise ValueError(f"unknown check '{example.check}'")


def _exact_problem(files: ExampleFiles, out: RunOutput) -> str | None:
    """The first difference between the expected outputs and those of the run, or None.

    Every expected output is in the run: ``_missing_output`` has checked it.
    """
    datasets = {dataset.name: dataset for dataset in out.datasets}
    scalars = {scalar.name: scalar for scalar in out.scalars}
    for wanted_dataset in files.expected_datasets:
        problem = compare_dataset(wanted_dataset, datasets[wanted_dataset.name])
        if problem is not None:
            return problem
    for wanted_scalar in files.expected_scalars:
        problem = compare_scalar(wanted_scalar, scalars[wanted_scalar.name])
        if problem is not None:
            return problem
    return None


def _random_problem(files: ExampleFiles, out: RunOutput) -> str | None:
    """Why the run of a ``random`` example is not a stable run of numbers in [0, 1), or None.

    The values are the non-null ones of the components that are not identifiers (identifiers
    are keys) and of the scalars. A run with no value does not show that anything is random.
    """
    if out.stable is None:
        return "not run twice, so not known to be stable"
    if not out.stable:
        return "the values change between runs"
    values = [value for role, _, value in _values(files, out) if role != _IDENTIFIER]
    if not values:
        return "no value to check"
    for value in values:
        if not _in_unit_interval(value):
            return f"{value!r} is not a number in [0, 1)"
    return None


def _current_date_problem(files: ExampleFiles, out: RunOutput, today: str) -> str | None:
    """Why the run of a ``current_date`` example has no Date value, or one that is not today."""
    dates = [value for _, vtype, value in _values(files, out) if vtype == "Date"]
    if not dates:
        return "no Date value"
    for value in dates:
        if not values_equal("Date", today, value):
            return f"{value} is not today, {today}"
    return None


def _in_unit_interval(text: str) -> bool:
    """Whether ``text`` is a number from 0 up to, but not including, 1."""
    text = text.strip()
    return _NUMBER.fullmatch(text) is not None and 0 <= float(text) < 1


def _expected_outputs(files: ExampleFiles, out: RunOutput) -> list[Output]:
    """The outputs of ``out`` that the example expects, datasets first: the rest is ignored."""
    datasets = {wanted.name for wanted in files.expected_datasets}
    scalars = {wanted.name for wanted in files.expected_scalars}
    return [
        *(dataset for dataset in out.datasets if dataset.name in datasets),
        *(scalar for scalar in out.scalars if scalar.name in scalars),
    ]


def _values(files: ExampleFiles, out: RunOutput) -> Iterator[tuple[str | None, str, str]]:
    """Every non-null value of the expected outputs of ``out``, as ``(role, type, value)``.

    The type is the canonical name of the VTL type. A scalar has no role: it is None.
    """
    for output in _expected_outputs(files, out):
        if isinstance(output, ActualScalar):
            if output.value is not None:
                yield None, normalize_type(output.type), output.value
            continue
        for name, component in output.components.items():
            vtype = normalize_type(component.type)
            for data_point in output.rows:
                value = data_point.get(name)
                if value is not None:
                    yield component.role, vtype, value


def _engines_agree(
    files: ExampleFiles,
    left: EngineVerdict,
    right: EngineVerdict,
    pandas: RunOutput,
    duckdb: RunOutput,
) -> bool:
    if left.outcome != right.outcome:
        return False
    if left.outcome == "error":
        return True
    return outputs_equal(_expected_outputs(files, pandas), _expected_outputs(files, duckdb))


def _status(left: EngineVerdict, right: EngineVerdict, agree: bool) -> Status:
    if not agree:
        return "diverge"
    if left.outcome == "match" and right.outcome == "match":
        return "ok"
    if left.outcome == "error" and right.outcome == "error":
        return "missing"
    return "partial"


# --- Roll-up ---


def rollup(statuses: Sequence[Status]) -> Status:
    """The status of a group, such as the examples of a variant or the variants of an operator.

    It is ``diverge`` if any member is, else ``ok`` if all are, else ``missing`` if all are,
    else ``partial``. An empty group has no status: it raises ``ValueError``.
    """
    if not statuses:
        raise ValueError("nothing to roll up: no statuses")
    if "diverge" in statuses:
        return "diverge"
    if all(status == "ok" for status in statuses):
        return "ok"
    if all(status == "missing" for status in statuses):
        return "missing"
    return "partial"


def counts(statuses: Sequence[Status]) -> dict[Status, int]:
    """How many of ``statuses`` are of each status: all four are keys, even with a zero."""
    tally = Counter(statuses)
    return {status: tally[status] for status in _STATUSES}


def build_index(cat: Catalogue, results: Mapping[str, ExampleResult]) -> StatusIndex:
    """The statuses of every example, variant, operator and chapter of ``cat``.

    ``results`` has the result of an example by its key: ``<chapter>/<operator>/<variant>/
    <example>``, or ``types/<type>/<variant>/<example>`` for an example of a type. A key of
    ``results`` that is not an example of the catalogue is ignored. An example of the catalogue
    with no result raises ``ValueError``, naming the first one in catalogue order, and so does
    an entry that has nothing to roll up, such as a variant with no examples. The statuses are
    rolled up with ``rollup``. A type with no variants has no status, and is not part of the
    ``types`` chapter, which is absent when no type has one.
    """
    examples: dict[str, ExampleResult] = {}
    variants: dict[str, Status] = {}
    operators: dict[str, Status] = {}
    chapters: dict[str, Status] = {}

    def roll_variants(owner: str, owned: Iterable[Variant]) -> list[Status]:
        """Index the variants of ``owner`` and their examples; the statuses of the variants."""
        statuses: list[Status] = []
        for variant in owned:
            variant_key = f"{owner}/{variant.id}"
            found: list[Status] = []
            for example in variant.examples:
                example_key = f"{variant_key}/{example.name}"
                if example_key not in results:
                    raise ValueError(f"no result for example {example_key}")
                examples[example_key] = results[example_key]
                found.append(results[example_key].status)
            variants[variant_key] = _rolled(variant_key, found)
            statuses.append(variants[variant_key])
        return statuses

    for chapter in cat.chapters:
        chapter_statuses: list[Status] = []
        for operator in chapter.operators:
            operator_key = f"{chapter.id}/{operator.id}"
            operators[operator_key] = _rolled(
                operator_key, roll_variants(operator_key, operator.variants)
            )
            chapter_statuses.append(operators[operator_key])
        chapters[chapter.id] = _rolled(chapter.id, chapter_statuses)

    type_statuses: list[Status] = []
    for typedef in cat.types:
        if typedef.variants:
            type_key = f"{_TYPES}/{typedef.id}"
            operators[type_key] = _rolled(type_key, roll_variants(type_key, typedef.variants))
            type_statuses.append(operators[type_key])
    if type_statuses:
        chapters[_TYPES] = rollup(type_statuses)
    return StatusIndex(examples, variants, operators, chapters)


def _rolled(key: str, statuses: Sequence[Status]) -> Status:
    """``rollup`` of the children of ``key``, which names the entry when it has none."""
    if not statuses:
        raise ValueError(f"nothing to roll up for {key}")
    return rollup(statuses)
