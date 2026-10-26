"""Verdicts of the engines on an example, the status they make, and the roll-up of statuses.

``verdict`` says what one engine gave for an example: ``match`` when it gave the expected
outputs, ``mismatch`` when it ran but gave something else, ``error`` when it did not run. A
failed run is one of two things, which ``is_guarded`` tells apart. A guarded error is an error
that vtlengine raised on purpose: one of its own classes (``VTL_ERRORS``), with a code. A crash
is any other failure: an unhandled exception such as ``AttributeError`` or ``KeyError``, or a
``Timeout`` or a ``Crash`` of the child. ``mark`` writes a verdict as a page shows it: ✅ for a
match, ❌ for a wrong result, 🛡 with its code for a guarded error, and 💥 with its type for a
crash.

An example can expect an error instead of outputs (``ExampleFiles.expected_error``): a run of
it matches when it raises a guarded error with exactly that code, and is a mismatch when it
raises another one, or none. A crash is a crash there too, never a match.

``example_result`` puts the verdicts of the two engines together into the status of the example:

* ``diverge``: the engines do not agree. Two engines agree when both fail, whatever their
  messages, or when they have the same verdict and ``outputs_equal`` finds their outputs equal.
  An example that expects an error has no outputs to compare: its engines agree when they raised
  the same code, or no error at all;
* ``ok``: both engines match;
* ``refused``: both engines fail with a guarded error, and neither code says that something is
  not implemented (``NOT_IMPLEMENTED``). Where the manual defines a result, the engines refuse
  it on purpose;
* ``missing``: both engines fail, and not in that way: with a crash, or with an error that says
  that something is not implemented;
* ``partial``: the engines agree and both give a wrong answer.

Only the outputs that the expected files of the example name are looked at, on both sides. A
script may return more, such as intermediate datasets: those are ignored. An expected output that
a run does not have is a mismatch. The examples that are checked by property (``random`` and
``current_date``) are judged by a property of their values, not by expected values, but their
outputs still have to have the expected structure. A ``random`` example may also expect values,
for the columns that are not draws, such as a seed that is kept: those are compared as in an
exact example, and the property is checked on the empty columns only. The engines still have to
give the same values, so ``random`` diverges while the engines use different generators.

``rollup`` carries the statuses up from examples to variants, operators and chapters, and
``build_index`` does it for a whole catalogue.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal

from tool.compare import (
    ActualDataset,
    ActualScalar,
    Output,
    compare_dataset,
    compare_scalar,
    normalize_role,
    normalize_type,
    outputs_equal,
    values_equal,
)
from tool.examples import ExampleFiles, ExpectedDataset, ExpectedScalar
from tool.model import Catalogue, Example, Status, Variant
from tool.runner import RunOutput

EMOJI: Final[dict[str, str]] = {
    "ok": "🟢",
    "partial": "🟡",
    "refused": "🛡️",
    "missing": "🔴",
    "diverge": "🔵",
}

Outcome = Literal["match", "mismatch", "error"]

# The classes of the errors that vtlengine raises on purpose. A failed run is a guarded error
# when it is of one of them and has a code; any other failure is a crash.
VTL_ERRORS: Final[frozenset[str]] = frozenset(
    {"SemanticError", "RunTimeError", "InputValidationException", "DataLoadError"}
)

# The codes of the errors that say that vtlengine does not implement something, each with its
# message as ``docs/error_messages.rst`` writes it, placeholders included. The list is kept by
# hand. A check of the tool builds the file of the messages from the catalogue of the engine, and
# compares, so that a message that changes there does not go unnoticed here.
NOT_IMPLEMENTED: Final[dict[str, str]] = {
    "1-3-13": "Unsupported VTL construct: {construct}.",
    "0-5-1-9": "Dataset Priority input mode on HR is not implemented",
    "1-3-2-2": "At op ``{op}``: User defined operator without returns is not implemented.",
    "1-3-2-5": (
        "At op eval: Eval in component execution is not supported, it only applies to Data Sets."
    ),
    "1-3-3-8": (
        "Viral propagation definition {name} is defined on value domain {value_domain}, "
        "which is not supported yet; define it on a variable instead."
    ),
    "0-5-1-8": (
        "Value Domain '{name}' not available for cast operator or scalar type representation "
        "or rulesets."
    ),
}

_STATUSES: Final[tuple[Status, ...]] = ("ok", "partial", "refused", "missing", "diverge")
# The statuses that a group has when all its members have it, in the order that ``rollup`` asks.
_UNIFORM: Final[tuple[Status, ...]] = ("ok", "refused", "missing")
_IDENTIFIER: Final = "Identifier"

# A decimal number in ASCII digits. ``float`` alone would also take ``nan``, ``inf`` and ``1_0``.
_NUMBER: Final = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)


@dataclass(frozen=True)
class EngineVerdict:
    """What an engine gave for an example; ``detail`` is empty for a ``match``.

    ``error_type`` and ``error_code`` are those of the error that the run raised, when it
    raised one: the class of the exception and, for an error of vtlengine, its code. A run that
    raised nothing has neither.
    """

    outcome: Outcome
    detail: str
    error_type: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class ExampleResult:
    """The verdicts of both engines on an example, whether they agree, and the status."""

    pandas: EngineVerdict
    duckdb: EngineVerdict
    engines_agree: bool
    status: Status


@dataclass(frozen=True)
class StatusIndex:
    """The statuses of a catalogue at every level, by key; the keys are in catalogue order."""

    examples: dict[str, ExampleResult]  # "chapter/op/variant/example"
    variants: dict[str, Status]  # "chapter/op/variant"
    operators: dict[str, Status]  # "chapter/op"
    chapters: dict[str, Status]  # "chapter"


# --- Verdicts ---


def verdict(example: Example, files: ExampleFiles, out: RunOutput, today: str) -> EngineVerdict:
    """What the run ``out`` of ``example`` on an engine gave: a match, a mismatch or an error.

    A run that failed is an ``error``, whose detail is ``<type> <code>: <message>`` with the
    parts that are missing left out; the type and the code are the ``error_type`` and the
    ``error_code`` of the verdict too.

    An example that expects an error (``files.expected_error``) is judged before anything else,
    and has no outputs to look at. A run that raised a guarded error with that code is a
    ``match``. One that raised another guarded error is a ``mismatch`` with the detail
    ``expected error <code>, got <code>``, and one that did not fail is a ``mismatch`` with the
    detail ``expected error <code>, ran without an error``. A failure that is not a guarded
    error is an ``error``, a crash.

    Otherwise every expected output must be in the run, by its name (an output that is not named
    in the expected files is ignored), or the verdict is a ``mismatch`` with the detail
    ``missing output <name>``. Then, by the ``check`` of the example:

    * ``exact``: every expected dataset and scalar must compare as equal, or the verdict is a
      ``mismatch`` and the detail is the first problem found, with the name of the output in
      front (``DS_s: ...``) when the example expects more than one;
    * ``random``: the run must be ``stable`` (the same on every run). A column of an expected
      dataset that has a value in its file is compared as in ``exact``, with the data points
      found by the identifiers (a file with no data point has no value to compare, and does not
      look at the identifiers). Every other column that is not an identifier, and every scalar, is
      a draw: its non-null values must be numbers from 0 up to, but not including, 1, and the run
      must have at least one. A problem names its output, as in ``exact``;
    * ``current_date``: there must be at least one Date value, and every non-null Date value,
      of a dataset or a scalar, must be ``today``.

    ``random`` and ``current_date`` look at the structure first. The expected datasets must have
    the components, roles and types that ``compare_dataset`` asks for (their data points are not
    compared) and the expected scalars their type (not their value), or the verdict is a
    ``mismatch`` with the first difference as detail.
    """
    if files.expected_error is not None:
        return _expected_error_verdict(files.expected_error, out)
    if not out.ok:
        return _error_verdict(out)
    problem = _missing_output(files, out)
    if problem is None:
        problem = _check_problem(example, files, out, today)
    if problem is None:
        return EngineVerdict("match", "")
    return EngineVerdict("mismatch", problem)


def is_guarded(v: EngineVerdict) -> bool:
    """Whether the verdict is about an error that vtlengine raised on purpose.

    That is an error of one of its own classes, ``VTL_ERRORS``, with a code. Any other failed
    run is a crash, whatever its type, a ``Timeout`` or a ``Crash`` included.
    """
    return v.error_type in VTL_ERRORS and v.error_code is not None


def mark(v: EngineVerdict) -> str:
    """The mark of a verdict, as a page shows it.

    ``✅`` is a match, ``❌`` is a wrong result, ``🛡 <code>`` is a guarded error with its code,
    and ``💥 <type>`` is a crash with the type of its exception. A crash that has no type is
    just ``💥``.
    """
    if v.outcome == "match":
        return "✅"
    if v.outcome == "mismatch":
        return "❌"
    if is_guarded(v):
        return f"🛡 {v.error_code}"
    return f"💥 {v.error_type}" if v.error_type else "💥"


def example_result(
    example: Example, files: ExampleFiles, pandas: RunOutput, duckdb: RunOutput, today: str
) -> ExampleResult:
    """The verdicts of ``example`` on both engines, whether the engines agree, and its status.

    The engines agree when both fail, and not when their verdicts differ. When both run to the
    same verdict they agree if the outputs that the example expects are equal on both
    (``outputs_equal``); an example that expects an error has none, and its engines agree if they
    raised the same code, or none. The status is ``diverge`` when they do not agree, ``ok`` when
    both match, and ``partial`` when both give the same wrong answer. When both fail it is
    ``refused`` if both errors are guarded and neither code is in ``NOT_IMPLEMENTED``, and
    ``missing`` otherwise.
    """
    left = verdict(example, files, pandas, today)
    right = verdict(example, files, duckdb, today)
    agree = _engines_agree(files, left, right, pandas, duckdb)
    return ExampleResult(left, right, agree, judge(left, right, agree))


def _error_detail(out: RunOutput) -> str:
    """``<type> <code>: <message>`` of a failed run, with what is missing left out."""
    head = " ".join(part for part in (out.error_type, out.error_code) if part)
    if head and out.error_message:
        return f"{head}: {out.error_message}"
    return head or out.error_message or ""


def _error_verdict(out: RunOutput) -> EngineVerdict:
    """The ``error`` verdict of a failed run, which keeps the type and the code of its error."""
    return EngineVerdict("error", _error_detail(out), out.error_type, out.error_code)


def _expected_error_verdict(expected: str, out: RunOutput) -> EngineVerdict:
    """The verdict of a run of an example that expects the error with the code ``expected``.

    Only a guarded error counts: a crash stays an ``error``, whatever it says.
    """
    if out.ok:
        return EngineVerdict("mismatch", f"expected error {expected}, ran without an error")
    raised = _error_verdict(out)
    if not is_guarded(raised):
        return raised
    if raised.error_code == expected:
        return replace(raised, outcome="match", detail="")
    detail = f"expected error {expected}, got {raised.error_code}"
    return replace(raised, outcome="mismatch", detail=detail)


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
    if example.check not in ("random", "current_date"):
        raise ValueError(f"unknown check '{example.check}'")
    # A property is about the values, but the expected files still say what the outputs are.
    problem = _structure_problem(files, out)
    if problem is not None:
        return problem
    if example.check == "random":
        return _random_problem(files, out)
    return _current_date_problem(files, out, today)


def _exact_problem(files: ExampleFiles, out: RunOutput) -> str | None:
    """The first difference between the expected outputs and those of the run, or None."""
    return _first_problem(files, out, compare_dataset, compare_scalar)


def _structure_problem(files: ExampleFiles, out: RunOutput) -> str | None:
    """The first difference in structure between the expected outputs and those of the run.

    A dataset must have the expected components, roles and types, by the rules of
    ``compare_dataset``, and a scalar its expected type. The values are left out.
    """
    return _first_problem(files, out, _dataset_structure, _scalar_type)


def _dataset_structure(expected: ExpectedDataset, actual: ActualDataset) -> str | None:
    """``compare_dataset`` without data points.

    With no data points on either side, only the structure is compared.
    """
    return compare_dataset(replace(expected, rows=[]), replace(actual, rows=[]))


def _scalar_type(expected: ExpectedScalar, actual: ActualScalar) -> str | None:
    """``compare_scalar`` without the value.

    The expected value is set to the one of the run, so that only the type can differ.
    """
    return compare_scalar(replace(expected, value=actual.value), actual)


def _first_problem(
    files: ExampleFiles,
    out: RunOutput,
    dataset_rule: Callable[[ExpectedDataset, ActualDataset], str | None],
    scalar_rule: Callable[[ExpectedScalar, ActualScalar], str | None],
) -> str | None:
    """The first problem that a rule finds between an expected output and the one of the run.

    The datasets come first, then the scalars, each in the order of the expected files. Every
    expected output is in the run: ``_missing_output`` has checked it. The problem names its
    output (``<name>: ...``) when there is more than one expected output, to tell them apart.
    """
    datasets = {dataset.name: dataset for dataset in out.datasets}
    scalars = {scalar.name: scalar for scalar in out.scalars}
    several = len(files.expected_datasets) + len(files.expected_scalars) > 1
    for wanted_dataset in files.expected_datasets:
        problem = dataset_rule(wanted_dataset, datasets[wanted_dataset.name])
        if problem is not None:
            return f"{wanted_dataset.name}: {problem}" if several else problem
    for wanted_scalar in files.expected_scalars:
        problem = scalar_rule(wanted_scalar, scalars[wanted_scalar.name])
        if problem is not None:
            return f"{wanted_scalar.name}: {problem}" if several else problem
    return None


def _random_problem(files: ExampleFiles, out: RunOutput) -> str | None:
    """Why the run of a ``random`` example is not a stable run with draws in [0, 1), or None.

    The columns of an expected dataset that have a value in the file are compared, and the
    others (the draws) are checked for numbers in [0, 1), as are the values of the scalars. A run
    with no draw does not show that anything is random.
    """
    if out.stable is None:
        return "not run twice, so not known to be stable"
    if not out.stable:
        return "the values change between runs"
    problem = _first_problem(files, out, _random_dataset, _random_scalar)
    if problem is not None:
        return problem
    if next(_draws(files, out), None) is None:
        return "no value to check"
    return None


def _random_dataset(expected: ExpectedDataset, actual: ActualDataset) -> str | None:
    """The first problem of a dataset of a ``random`` example: its values, then its draws.

    A file with no data point holds no value: only the draws are checked then, as they all are.
    """
    draws = _draw_columns(expected)
    if expected.rows:
        kept = {name: each for name, each in expected.components.items() if name not in draws}
        problem = compare_dataset(
            replace(expected, components=kept),
            replace(actual, components={name: actual.components[name] for name in kept}),
        )
        if problem is not None:
            return problem
    return _bad_draw(_dataset_draws(draws, actual))


def _random_scalar(expected: ExpectedScalar, actual: ActualScalar) -> str | None:
    """The problem of a scalar of a ``random`` example: it is a draw, whatever the file says."""
    return _bad_draw(_scalar_draws(actual))


def _draw_columns(expected: ExpectedDataset) -> list[str]:
    """The columns of an expected dataset that are draws: not identifiers, and empty in the file.

    Empty is no value in any data point, so every column that is not an identifier is a draw when
    the file has no data point at all.
    """
    return [
        name
        for name, component in expected.components.items()
        if normalize_role(component.role) != _IDENTIFIER
        and all(not (data_point.get(name) or "").strip() for data_point in expected.rows)
    ]


def _dataset_draws(columns: Sequence[str], actual: ActualDataset) -> Iterator[str]:
    """The non-null values that the dataset the run gave has in the draw ``columns``."""
    for name in columns:
        for data_point in actual.rows:
            value = data_point.get(name)
            if value is not None:
                yield value


def _scalar_draws(actual: ActualScalar) -> Iterator[str]:
    """The value of a scalar the run gave, if it is not null."""
    if actual.value is not None:
        yield actual.value


def _draws(files: ExampleFiles, out: RunOutput) -> Iterator[str]:
    """Every non-null draw of the expected outputs of the run."""
    datasets = {dataset.name: dataset for dataset in out.datasets}
    scalars = {scalar.name: scalar for scalar in out.scalars}
    for wanted_dataset in files.expected_datasets:
        columns = _draw_columns(wanted_dataset)
        yield from _dataset_draws(columns, datasets[wanted_dataset.name])
    for wanted_scalar in files.expected_scalars:
        yield from _scalar_draws(scalars[wanted_scalar.name])


def _bad_draw(values: Iterable[str]) -> str | None:
    """The problem of the first of ``values`` that is not a number in [0, 1), or None."""
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
    """Whether the engines agree: they do when both fail, and not when their outcomes differ.

    When both run to the same outcome, the outputs that the example expects must be equal on
    both. An example that expects an error has no outputs: its engines agree when they raised
    the same code, which is the expected one when both match, or no code when neither failed.
    """
    if left.outcome != right.outcome:
        return False
    if left.outcome == "error":
        return True
    if files.expected_error is not None:
        return left.error_code == right.error_code
    return outputs_equal(_expected_outputs(files, pandas), _expected_outputs(files, duckdb))


def judge(left: EngineVerdict, right: EngineVerdict, agree: bool) -> Status:
    """The status of an example from the verdicts of its engines, and whether they agree.

    It is ``diverge`` when the engines do not agree, and ``ok`` when both match. When both fail
    it is ``refused`` if both errors are guarded and neither code says that something is not
    implemented, and ``missing`` if either is a crash or such an error. Otherwise the engines
    agree on a wrong answer, and it is ``partial``. ``example_result`` uses it, and so does the
    reading of the ``status.json`` of an earlier run, which judges its verdicts again.
    """
    if not agree:
        return "diverge"
    if left.outcome == "match" and right.outcome == "match":
        return "ok"
    if left.outcome == "error" and right.outcome == "error":
        return "refused" if _refuses(left) and _refuses(right) else "missing"
    return "partial"


def _refuses(v: EngineVerdict) -> bool:
    """Whether a failure is an error that vtlengine raises on purpose, not a gap or a crash."""
    return is_guarded(v) and v.error_code not in NOT_IMPLEMENTED


# --- Roll-up ---


def rollup(statuses: Sequence[Status]) -> Status:
    """The status of a group, such as the examples of a variant or the variants of an operator.

    It is ``diverge`` if any member is, else ``ok`` if all are, else ``refused`` if all are, else
    ``missing`` if all are, else ``partial``. An empty group has no status: it raises
    ``ValueError``.
    """
    if not statuses:
        raise ValueError("nothing to roll up: no statuses")
    if "diverge" in statuses:
        return "diverge"
    for uniform in _UNIFORM:
        if all(status == uniform for status in statuses):
            return uniform
    return "partial"


def counts(statuses: Sequence[Status]) -> dict[Status, int]:
    """How many of ``statuses`` are of each status: all five are keys, even with a zero."""
    tally = Counter(statuses)
    return {status: tally[status] for status in _STATUSES}


def build_index(cat: Catalogue, results: Mapping[str, ExampleResult]) -> StatusIndex:
    """The statuses of every example, variant, operator and chapter of ``cat``.

    ``results`` has the result of an example by its key, ``<chapter>/<operator>/<variant>/
    <example>``. A key of ``results`` that is not an example of the catalogue is ignored. An
    example of the catalogue with no result raises ``ValueError``, naming the first one in
    catalogue order, and so does an entry that has nothing to roll up, such as a variant with no
    examples. The statuses are rolled up with ``rollup``.
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
    return StatusIndex(examples, variants, operators, chapters)


def _rolled(key: str, statuses: Sequence[Status]) -> Status:
    """``rollup`` of the children of ``key``, which names the entry when it has none."""
    if not statuses:
        raise ValueError(f"nothing to roll up for {key}")
    return rollup(statuses)
