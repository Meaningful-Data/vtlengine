"""Generate the operator map: validate, run every example on both engines, judge, write.

    generate.py [--only CHAPTER]... [--workers 4] [--timeout 120] [--check-only | --compare]
                [--summary FILE] [--fresh-processes]

It needs a Python that can import vtlengine, with the compiled parser in src/ ("Building the C++
parser from source" in CONTRIBUTING.md says how to build it): every example runs in a child
process that imports the engines of this worktree, and one child checks that it can before any
example runs. Where systemd is available, run it under a memory cap, from the repository root:

systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 python docs/operator_map/generate.py

Elsewhere the cap is optional.

1. The catalogue is checked against the index of the manual (``validate_catalogue``), and the
   example folders against the catalogue in both directions: a catalogue example needs a folder
   that loads, with a ``notes.rst`` in the fixed layout and no ``notes.md``
   (``tool.examples.notes_problems``), and a folder needs a catalogue example. The forms must
   declare the types they take and give (``any`` is only for what a form gives), the forms of an
   operator must all have a kind or none (unless they name boxes) and all name a light box or
   none (``Variant.box``, with a title of 1 to 30 characters), their authored examples must
   be named after what they test, and no two forms or examples of an operator may have the same
   anchor on its page (``form_anchor``, ``example_anchor``). No link of the catalogue goes from
   or to a definition of the VTL-DL.
2. One child process imports vtlengine and both engines, and gives their versions (the
   preflight of ``tool.runner``). Then every example runs on the pandas and the DuckDB engine,
   each in a process of its own, forked from a server that imported the engines once (with
   ``--fresh-processes``, or where the platform cannot fork, a fresh interpreter each). A
   ``random`` example runs twice, to see whether its values are stable. A line every 5 seconds
   says how far the run has got.
3. The status of each example comes from the two runs (``tool.status``), the statuses are rolled
   up, and the files are rendered: ``index.rst``, how to read (``how_to_read.rst``), the
   overview (``overview.rst``), the page of the blocks of operators (``operators.rst``), the team
   decisions, errata and adaptations (``decisions.rst``), a page for each block
   (``<chapter>/index.rst``)
   and for each operator (``<chapter>/<operator>.rst``), ``status.json`` and
   ``example_pages.json``, which says for each example its title, status, marks and where it is
   on the page of its operator. The pages are documents of the Sphinx docs of the repository
   (``docs/``), which copies the examples next to them when it builds the site.
4. Only when every file has been rendered, and every diagram is within the limits of Mermaid, are
   the files written: first to a folder of their own, then moved over the old ones, in the
   folders of the blocks, which are made as needed. The manifest ``.generated`` lists the files
   of the run, one path per line, sorted, and is read by the next run: a file that it lists and
   the next run does not generate (the page of an operator that was renamed or removed, or of a
   block that is gone) is removed, with its folder if it is left empty. A file that it does not
   list is never touched, and nor is a path that it lists outside the generated pages: a line
   that is not the name of a generated file, or that goes through a symbolic link, is a problem
   (``_read_manifest``). The manifest is written last, after the removals.

With ``--only`` just the named chapters run. The other chapters keep their results from the
``status.json`` of the previous run: an example that is not run and has no result there is a
problem. That file is read only when a chapter with examples is left out of ``--only``. Every run
renders every page, from the folders of every chapter, so the checks of step 1 cover every
chapter whatever ``--only`` says: it limits which examples run, and the page coverage and the
manual example mapping to the pages of the chapters named. Nothing is run or written with
``--check-only``: it stops after the checks, and does not ask for previous results.

With ``--compare`` the examples run as always (all of them, or the chapters of ``--only``), and
their statuses are compared with the committed ``status.json`` (``tool.baseline``): nothing is
written. A status that ranks lower than there is a regression; the examples that regressed run a
second time, and a regression counts only if it is there both times. The improvements, the
changes between 🟢 and 🛡️, and the new and removed examples are reported, and never fail. The
exit code is 1 with a regression, and 0 otherwise. ``--summary FILE`` appends the same report to
``FILE`` as Markdown (the summary of a CI job): with ``--compare``, against the committed file;
in a run that writes, against the ``status.json`` that the run replaces.

The exit code is 0 for a run that wrote its files, 1 for any problem (printed one per line, and
nothing is written), and 2 for a run that was interrupted, wherever the interrupt came, that hit
an internal error (one line: its type and message), or whose preflight child cannot import the
engines (one line: the reason; nothing is run or written). Until every new file is in place, a
failure or an interrupt leaves the files of the previous run as they were: the files that were
moved are put back, the stale files that were removed are put back too, the old manifest is
still the manifest, and the folders that the run made are removed. Only when putting one back
fails too, even by a second interrupt, the folder that has the old bytes is kept, and the line
says where it is. After that point nothing can be undone: an interrupt or an error in the
clean-up or in the summary leaves the new files, and the line says so, and names a
``.generate-*`` folder that the clean-up did not remove. This process never imports vtlengine or
pandas: the engines run in the children of ``tool.runner``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, get_args

from catalogue import manual_index
from tool.baseline import compare, report_lines, summary_markdown
from tool.examples import (
    ExampleError,
    ExampleFiles,
    example_dir,
    load_example,
    notes_problems,
    scan_example_dirs,
)
from tool.mermaid import MermaidLimitError, check_limits, example_anchor, form_anchor
from tool.model import (
    Catalogue,
    Example,
    Operator,
    Status,
    example_title,
    load_catalogue,
    validate_catalogue,
)
from tool.render import (
    RUN_FILE,
    RunMeta,
    block_rst,
    decisions_rst,
    how_to_read_rst,
    index_rst,
    mermaid_blocks,
    operator_rst,
    operators_rst,
    overview_rst,
    run_html,
)
from tool.runner import Engine, EngineVersions, PreflightError, RunOutput, preflight, run_all
from tool.status import (
    EMOJI,
    EngineVerdict,
    ExampleResult,
    Outcome,
    StatusIndex,
    build_index,
    counts,
    example_result,
    judge,
    mark,
)

ROOT = Path(__file__).resolve().parent  # docs/operator_map: where the files are written
REPO = ROOT.parents[1]  # the worktree: its git commit and vtlengine version are the engine's

_STATUS_FILE: Final = "status.json"
_EXAMPLE_PAGES_FILE: Final = "example_pages.json"
# The manifest: the paths that a run generated, one on each line, sorted. It is not one of them.
_MANIFEST: Final = ".generated"
# The paths that a run generates: the index, ``status.json`` and ``example_pages.json`` in the
# folder of the map, and the pages of a block in the folder of its chapter, all named by the rule
# of the ids of the catalogue. The folders that people keep in the map are never a block's.
_GENERATED_PATH: Final = re.compile(r"[a-z0-9_]+\.(?:rst|json|html)|[a-z0-9_]+/[a-z0-9_]+\.rst")
_KEPT_FOLDERS: Final = frozenset({"catalogue", "examples", "tool"})
# The folders of the staging folder: the new files, as they will be in the folder of the map, and
# the old bytes of every file that the move replaces or removes. Two folders, so that no name of
# a generated page can be the name of the other.
_NEW: Final = "new"
_OLD: Final = "old"
_MAX_UNRUN: Final = 10  # the "no result" problems that are printed, one line each
_ENGINES: Final[tuple[Engine, ...]] = ("pandas", "duckdb")
_VERSION: Final = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.MULTILINE)
# The head of the detail of a failed run: ``<type> <code>: <message>`` with the code optional, or
# only ``<type> <code>`` or ``<type>`` for a run that said no message. A ``status.json`` of the
# tool before the fields ``error_type`` and ``error_code`` says them so.
_ERROR_HEAD: Final = re.compile(r"^(\w+)(?: (\d+(?:-\d+)+))?(?::|$)")
_INDEX_FILE: Final = "index.rst"
_HOW_TO_READ_FILE: Final = "how_to_read.rst"
_OVERVIEW_FILE: Final = "overview.rst"
_OPERATORS_FILE: Final = "operators.rst"
_DECISIONS_FILE: Final = "decisions.rst"
# How to run the generator, for ``--help``; the docstring and the index say the same.
_HOW_TO_RUN: Final = """\
It needs a Python that can import vtlengine, with the compiled parser in src/
(see "Building the C++ parser from source" in CONTRIBUTING.md).

Where systemd is available, run it under a memory cap, from the repository root:

  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 python docs/operator_map/generate.py

Elsewhere the cap is optional."""


class GenerateError(Exception):
    """A problem that stops the run before anything is written, said in one line."""


class RestoreError(Exception):
    """A write failed and some old files could not be put back: the staging folder keeps them."""


@dataclass
class _Progress:
    """How far the write of the files has got, for the line of a run that stops.

    ``written`` says that every new file is in place, after which nothing can undo it, and
    ``leftover`` is a staging folder that exists and is meant to be removed.
    """

    written: bool = False
    leftover: Path | None = None


@dataclass(frozen=True)
class _Item:
    """One example of the catalogue, with where it lives."""

    group: str  # the chapter id
    operator: str  # the operator id
    variant: str
    example: Example

    @property
    def parts(self) -> tuple[str, str, str, str]:
        return self.group, self.operator, self.variant, self.example.name

    @property
    def key(self) -> str:
        return "/".join(self.parts)


# --- Command line ---


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generator; see the module docstring. Returns the exit code."""
    args = _parse_args(argv)
    progress = _Progress()
    try:
        return _generate(args, progress)
    except KeyboardInterrupt:  # wherever it comes: the line says what the files are
        print(_interrupted(progress), file=sys.stderr)
        return 2
    except Exception as exc:  # whatever it is, one line and exit 2
        print(_internal_error(exc, progress), file=sys.stderr)
        return 2


def _interrupted(progress: _Progress) -> str:
    """The line of an interrupt: the files are as they were until every new file is in place."""
    if progress.written:
        head = "interrupted after the files were written: every new file is in place"
    else:
        head = "interrupted: nothing was written"
    return head + _leftover(progress)


def _internal_error(exc: Exception, progress: _Progress) -> str:
    """The line of an internal error: its type and message, and what is on disk if it is new."""
    line = f"internal error: {type(exc).__name__}: {' '.join(str(exc).split())}"
    if progress.written:
        line += "; every new file is in place"
    return line + _leftover(progress)


def _leftover(progress: _Progress) -> str:
    """The end of a line that names the staging folder that the clean-up did not remove."""
    if progress.leftover is None:
        return ""
    return f"; the folder {progress.leftover} was left over, remove it"


def _generate(args: argparse.Namespace, progress: _Progress) -> int:
    """The run of ``main`` after the arguments: the exit codes 0 and 1."""
    cat = load_catalogue()
    only: list[str] = list(dict.fromkeys(args.only))
    known = [chapter.id for chapter in cat.chapters]
    if any(name not in known for name in only):
        return _fail(
            [
                f"unknown chapter '{name}' for --only (known: {', '.join(known)})"
                for name in only
                if name not in known
            ]
        )

    items = _items(cat)
    writes = not (args.check_only or args.compare)
    checked = _check(cat, items, only, ROOT, with_previous=writes)
    if checked.problems:
        return _fail(checked.problems)
    if args.check_only:
        print(f"ok: {len(items)} examples checked, nothing was run")
        return 0
    todo = [item for item in items if not only or item.group in only]
    baseline: dict[str, ExampleResult] = {}
    unread = ""  # why the summary of a run that writes has no baseline to compare with
    if args.compare:
        try:
            baseline = _baseline(only)
        except GenerateError as exc:
            return _fail([str(exc)])
    elif args.summary is not None:
        try:
            baseline = _load_previous(ROOT / _STATUS_FILE)
        except GenerateError as exc:
            unread = str(exc)
    try:
        versions = preflight(args.timeout)
    except PreflightError as exc:  # every run would crash: the pages must not take that
        print(str(exc), file=sys.stderr)
        return 2
    try:
        meta = _run_meta(versions)
    except GenerateError as exc:
        return _fail([str(exc)])

    print(f"running {len(todo)} examples on both engines ({args.workers} at a time)", flush=True)
    results = _run(todo, checked.loaded, args, meta)
    if args.compare:
        return _compare(baseline, results, todo, checked.loaded, args, meta)
    index = build_index(cat, {**checked.previous, **results})
    try:
        files = _render(cat, index, meta, ROOT / "examples")
    except MermaidLimitError as exc:
        return _fail([str(exc)])
    removed = _write_all(ROOT, files, progress)
    summary = " · ".join(
        f"{EMOJI[status]} {number}"
        for status, number in counts([each.status for each in index.examples.values()]).items()
    )
    pages = sum(1 for name in files if name.endswith(".rst"))
    print(
        f"wrote {pages} pages, {_STATUS_FILE}, {_EXAMPLE_PAGES_FILE} and {_MANIFEST}; "
        f"removed {len(removed)} stale files. Examples: {summary}"
    )
    if args.summary is not None:
        report = summary_markdown(compare(baseline, index.examples), index.examples, meta)
        if unread:
            report += f"\n{unread}: every example is reported as new.\n"
        _append_summary(args.summary, report)
    return 0


def _run(
    todo: Sequence[_Item],
    loaded: Mapping[str, ExampleFiles],
    args: argparse.Namespace,
    meta: RunMeta,
) -> dict[str, ExampleResult]:
    """Run ``todo`` on both engines and judge the runs: the result of each, by key."""
    examples_root = ROOT / "examples"
    progress = _RunProgress(examples_root)
    jobs = _jobs(todo, examples_root)
    outputs = run_all(jobs, args.workers, args.timeout, progress, fresh=args.fresh_processes)
    return _judge(todo, loaded, outputs, examples_root, meta.date)


# --- How far a run has got ---

_PROGRESS_EVERY_S: Final = 5.0  # a line at most this often, and one at the end


def _now() -> float:
    return time.monotonic()


class _RunProgress:
    """Prints how far a run has got, every ``_PROGRESS_EVERY_S`` seconds and when it ends.

    One line each time, never a line that rewrites itself: the lines are meant for the log of a
    CI job as much as for a terminal.
    """

    def __init__(self, examples_root: Path) -> None:
        self._root = examples_root
        self._start = _now()
        self._last = self._start

    def __call__(self, done: int, total: int, key: tuple[Path, Engine]) -> None:
        now = _now()
        if done < total and now - self._last < _PROGRESS_EVERY_S:
            return
        self._last = now
        path, engine = key
        try:
            example = path.relative_to(self._root).as_posix()
        except ValueError:
            example = path.name
        line = _progress_line(done, total, now - self._start, f"{example} on {engine}")
        print(line, flush=True)


def _progress_line(done: int, total: int, elapsed_s: float, last: str) -> str:
    """``[ 37%] 700/1888 runs · 3m12s elapsed · about 5m25s left · <the last run>``.

    The time left assumes that the runs to come take as long as those that ended. The last line
    of a run is only how many ran and how long it took.
    """
    head = f"[{done * 100 // total:3d}%] {done}/{total} runs · {_duration(elapsed_s)}"
    if done == total:
        return head
    left = elapsed_s / done * (total - done)
    return f"{head} elapsed · about {_duration(left)} left · {last}"


def _duration(seconds: float) -> str:
    """``31s``, ``3m12s`` or ``1h18m``."""
    whole = int(seconds)
    if whole < 60:
        return f"{whole}s"
    if whole < 3600:
        return f"{whole // 60}m{whole % 60:02d}s"
    return f"{whole // 3600}h{whole % 3600 // 60:02d}m"


# --- Comparing with the committed baseline ---


def _baseline(only: Sequence[str]) -> dict[str, ExampleResult]:
    """The results of the committed ``status.json``, of the chapters of ``only`` if it names any.

    Raises ``GenerateError`` when there is no file, or when it cannot be read.
    """
    path = ROOT / _STATUS_FILE
    if not path.is_file():
        raise GenerateError(f"{_STATUS_FILE}: no baseline to compare with (run generate.py first)")
    baseline = _load_previous(path)
    if only:
        baseline = {key: each for key, each in baseline.items() if key.split("/")[0] in only}
    return baseline


def _compare(
    baseline: Mapping[str, ExampleResult],
    results: dict[str, ExampleResult],
    todo: Sequence[_Item],
    loaded: Mapping[str, ExampleFiles],
    args: argparse.Namespace,
    meta: RunMeta,
) -> int:
    """Compare ``results`` with ``baseline``, print and append the report: 1 with a regression.

    The examples that regressed run a second time: a regression counts only if it is there both
    times, so that a run slowed down by a busy machine does not fail the check.
    """
    found = compare(baseline, results)
    regressed = {change.key for change in found.regressions}
    if regressed:
        again = _run([item for item in todo if item.key in regressed], loaded, args, meta)
        results.update(again)
        found = compare(baseline, results)
    for line in report_lines(found):
        print(line)
    passing = sorted(regressed - {change.key for change in found.regressions})
    if passing:
        noun = "regression" if len(passing) == 1 else "regressions"
        print(f"{len(passing)} {noun} did not repeat on a second run: {', '.join(passing)}")
    if args.summary is not None:
        _append_summary(args.summary, summary_markdown(found, results, meta))
    return 1 if found.regressions else 0


def _append_summary(path: Path, text: str) -> None:
    """Append ``text`` to the file at ``path``; a file that cannot be written is only said."""
    try:
        with path.open("a", encoding="utf-8") as summary:
            summary.write(text)
    except OSError as exc:
        print(f"cannot append the summary to {path}: {exc}", file=sys.stderr)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate the VTL operator map: run every example on both engines, "
        "judge the results and write the Sphinx pages (rst), status.json and example_pages.json.",
        epilog=_HOW_TO_RUN,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="CHAPTER",
        help="run only this chapter, and keep the results of the others from the previous run; "
        "repeat it to name several chapters. The catalogue and the example folders of every "
        "chapter are checked either way",
    )
    parser.add_argument(
        "--workers",
        type=_positive(int),
        default=4,
        help="how many examples run at a time (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive(float),
        default=120.0,
        help="seconds that a run of an example can take (default: %(default)s)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="check the catalogue and the example folders, then stop: nothing is run or written",
    )
    mode.add_argument(
        "--compare",
        action="store_true",
        help=f"run the examples and compare their statuses with the committed {_STATUS_FILE}, "
        "writing nothing: the exit code is 1 when an example ranks lower than there (a "
        "regression, which is run a second time to be sure), 0 otherwise",
    )
    parser.add_argument(
        "--fresh-processes",
        action="store_true",
        help="start a fresh interpreter for every run, instead of forking it from a server that "
        "imported the engines once: several times slower, for when the forked runs are in doubt",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        metavar="FILE",
        help=f"append to FILE a Markdown report of how the statuses moved from {_STATUS_FILE} "
        "(the one that the run replaces, or that --compare reads), for the summary of a CI job",
    )
    return parser.parse_args(argv)


def _positive(kind: type[int] | type[float]) -> Any:
    """An argparse type: a ``kind`` of number that is above zero."""

    def parse(text: str) -> Any:
        try:
            number = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if not 0 < number < math.inf:
            raise argparse.ArgumentTypeError(f"must be a finite number above zero: {text}")
        return number

    return parse


def _fail(problems: Sequence[str]) -> int:
    """Print the problems, one per line, and give the exit code of a run with problems."""
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1


# --- Checks ---


@dataclass(frozen=True)
class _Checked:
    """What the checks found: the problems, and what a run needs when there are none."""

    problems: list[str]
    loaded: dict[str, ExampleFiles]  # every example, by key: the ones that are to run are in it
    previous: dict[str, ExampleResult]  # the results of the previous run, with ``--only``


def _check(
    cat: Catalogue,
    items: Sequence[_Item],
    only: Sequence[str],
    root: Path,
    *,
    with_previous: bool,
) -> _Checked:
    """Check the catalogue and the example folders, and read the previous results if wanted.

    Every chapter is checked, whatever ``only`` says: a run renders the pages of all of them.
    The previous results are read only when ``only`` leaves out a chapter that has examples, and
    an example that is not run and has none is a problem. They are not needed to check, so
    ``with_previous`` leaves them out. It leaves out too the manifest of the previous run, which
    a run that writes reads, and a line of it that is not the path of a generated page is a
    problem here, before anything runs.
    """
    problems = _catalogue_problems(cat, only)
    folder_problems, loaded = _folder_problems(items, root)
    problems.extend(folder_problems)
    if with_previous:
        try:
            _read_manifest(root)
        except GenerateError as exc:
            problems.append(str(exc))
    previous: dict[str, ExampleResult] = {}
    if with_previous and _needs_previous(items, only):
        try:
            previous = _load_previous(root / _STATUS_FILE)
        except GenerateError as exc:
            problems.append(str(exc))
        else:
            problems.extend(_unrun_problems(items, only, previous))
    return _Checked(problems, loaded, previous)


def _needs_previous(items: Sequence[_Item], only: Sequence[str]) -> bool:
    """Whether the run leaves out a chapter that has examples, so that it needs its results."""
    return bool(only) and any(item.group not in only for item in items)


def _items(cat: Catalogue) -> list[_Item]:
    """Every example of the catalogue, in its order."""
    return [
        _Item(chapter.id, operator.id, variant.id, example)
        for chapter in cat.chapters
        for operator in chapter.operators
        for variant in operator.variants
        for example in variant.examples
    ]


def _catalogue_problems(cat: Catalogue, only: Sequence[str]) -> list[str]:
    """The problems of the catalogue against the manual, and what cannot be rolled up.

    ``only`` limits the coverage of the pages and of the manual examples to its chapters, and
    nothing else.
    """
    problems = validate_catalogue(
        cat, manual_index.MANUAL_PAGES, manual_index.MANUAL_EXAMPLES, scope=only or None
    )
    for chapter in cat.chapters:
        if not chapter.operators:
            problems.append(f"{chapter.id}: no operator")
        if chapter.id in _KEPT_FOLDERS:
            problems.append(
                f"{chapter.id}: its pages would go in the folder {chapter.id}/ of the map"
            )
        for operator in chapter.operators:
            if f"{operator.id}.rst" == _INDEX_FILE:
                problems.append(
                    f"{chapter.id}.{operator.id}: its page would be the page of the block, "
                    f"{chapter.id}/{_INDEX_FILE}"
                )
            if not operator.variants:
                problems.append(f"{chapter.id}.{operator.id}: no variant")
            problems.extend(
                f"{chapter.id}.{operator.id}.{variant.id}: no example"
                for variant in operator.variants
                if not variant.examples
            )
            problems.extend(_anchor_problems(f"{chapter.id}.{operator.id}", operator))
    return list(dict.fromkeys(problems))


def _anchor_problems(owner: str, operator: Operator) -> Iterator[str]:
    """The anchors that two forms or examples of ``operator`` share on its page.

    The anchors are ids of one HTML page (``form_anchor``, ``example_anchor``), and the
    underscores of the ids are dashes in them: the example ``c`` of the form ``a_b`` and the
    example ``b_c`` of the form ``a`` are both ``ex-a-b-c``. A shared id sends the click of a
    diagram, the links of a block or of the index and the back link of the page of an example to
    the wrong one. One line for each shared anchor, naming everything that has it. A form or an
    example that is declared twice is a duplicate that ``validate_catalogue`` reports: it is not
    named twice, and shares its anchor with no one else.
    """
    owners: dict[str, list[str]] = {}
    for variant in operator.variants:
        owners.setdefault(form_anchor(variant), []).append(f"form '{variant.id}'")
        for example in variant.examples:
            name = f"example '{variant.id}/{example.name}'"
            owners.setdefault(example_anchor(variant, example), []).append(name)
    for anchor, names in owners.items():
        distinct = list(dict.fromkeys(names))
        if len(distinct) > 1:
            yield f"{owner}: the anchor '{anchor}' is shared by {', '.join(distinct)}"


def _folder_problems(
    items: Sequence[_Item], root: Path
) -> tuple[list[str], dict[str, ExampleFiles]]:
    """The folders of the examples against the catalogue, and the examples that were loaded.

    A catalogue example needs a folder that loads, with its notes, and a folder with no catalogue
    example is a problem. This is so for every chapter: every run renders the pages of all of
    them from their folders, whichever chapters it runs.
    """
    examples_root = root / "examples"
    on_disk = scan_example_dirs(examples_root)
    problems: list[str] = []
    loaded: dict[str, ExampleFiles] = {}
    for item in items:
        path = example_dir(examples_root, *item.parts)
        if not path.is_dir():
            problems.append(f"missing example folder: {path.relative_to(root).as_posix()}")
            continue
        try:
            loaded[item.key] = load_example(path)
        except ExampleError as exc:
            problems.append(str(exc))
        problems.extend(notes_problems(path, item.example))
    wanted = {item.parts for item in items}
    for orphan in sorted(on_disk - wanted):
        path = example_dir(examples_root, *orphan)
        problems.append(f"example folder not in the catalogue: {path.relative_to(root).as_posix()}")
    return problems, loaded


def _unrun_problems(
    items: Sequence[_Item], only: Sequence[str], previous: Mapping[str, ExampleResult]
) -> list[str]:
    """The examples that are not run now and have no result from a previous run.

    One problem for each of the first ``_MAX_UNRUN``, then a line that counts the rest.
    """
    lines = [
        f"no result for {item.key}: its chapter is not run now and {_STATUS_FILE} has none "
        "(run its chapter too)"
        for item in items
        if item.group not in only and item.key not in previous
    ]
    if len(lines) > _MAX_UNRUN:
        return [*lines[:_MAX_UNRUN], f"… and {len(lines) - _MAX_UNRUN} more"]
    return lines


# --- Running ---


def _jobs(todo: Sequence[_Item], examples_root: Path) -> list[tuple[Path, Engine, int]]:
    """Each example on each engine, one after the other; a ``random`` example runs twice."""
    return [
        (
            example_dir(examples_root, *item.parts),
            engine,
            2 if item.example.check == "random" else 1,
        )
        for item in todo
        for engine in _ENGINES
    ]


def _judge(
    todo: Sequence[_Item],
    loaded: Mapping[str, ExampleFiles],
    outputs: Mapping[tuple[Path, Engine], RunOutput],
    examples_root: Path,
    today: str,
) -> dict[str, ExampleResult]:
    """The result of each example that ran: its status from the runs on both engines."""
    results: dict[str, ExampleResult] = {}
    for item in todo:
        path = example_dir(examples_root, *item.parts)
        pandas, duckdb = outputs[(path, "pandas")], outputs[(path, "duckdb")]
        results[item.key] = example_result(item.example, loaded[item.key], pandas, duckdb, today)
    return results


def _today() -> str:
    return date.today().isoformat()


def _run_meta(versions: EngineVersions) -> RunMeta:
    return RunMeta(
        date=_today(),
        vtlengine_commit=_engine_commit(REPO),
        vtlengine_version=_engine_version(REPO),
        spec_commit=manual_index.MANUAL_COMMIT,
        pandas_version=versions.pandas,
        duckdb_version=versions.duckdb,
    )


def _engine_commit(repo: Path) -> str:
    """The short commit of the worktree, from ``git rev-parse``."""
    try:
        done = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GenerateError(f"cannot run git in {repo}: {exc}") from exc
    commit = done.stdout.strip()
    if done.returncode != 0 or not commit:
        raise GenerateError(f"cannot read the commit of {repo}: {done.stderr.strip()}")
    return commit


def _engine_version(repo: Path) -> str:
    """The ``__version__`` of vtlengine, read from its source: this process never imports it."""
    path = repo / "src" / "vtlengine" / "__init__.py"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GenerateError(f"cannot read the version of vtlengine in {path}: {exc}") from exc
    found = _VERSION.search(text)
    if found is None:
        raise GenerateError(f'no __version__ = "..." in {path}')
    return found.group(1)


# --- The previous run ---


def _load_previous(path: Path) -> dict[str, ExampleResult]:
    """The results of the previous run, by example key; none if there is no file."""
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        return {key: _parse_result(entry) for key, entry in document["examples"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise GenerateError(
            f"{path.name}: cannot read the previous results ({type(exc).__name__}: {exc})"
        ) from exc


def _parse_result(raw: Any) -> ExampleResult:
    """The result of an example from its entry of ``status.json``.

    The status that the file holds is only checked to be one: the rules that make a status change
    from one version of the tool to the next (an error that the engines raise on purpose is
    ``refused``, where it was ``missing``), so the status is worked out again from the verdicts.
    """
    status, agree = raw["status"], raw["engines_agree"]
    if status not in get_args(Status) or not isinstance(agree, bool):
        raise ValueError(f"not a result: {raw!r}")
    pandas, duckdb = _parse_verdict(raw["pandas"]), _parse_verdict(raw["duckdb"])
    return ExampleResult(pandas, duckdb, agree, judge(pandas, duckdb, agree))


def _parse_verdict(raw: Any) -> EngineVerdict:
    """The verdict of an engine from its entry of ``status.json``.

    A file of the tool before the fields ``error_type`` and ``error_code`` has neither: they are
    read from the detail of an ``error``, which started with them. Nothing else has them to read:
    the detail of a ``mismatch`` is a problem of the outputs, which may start like an error does.
    """
    outcome, detail = raw["outcome"], raw["detail"]
    if outcome not in get_args(Outcome) or not isinstance(detail, str):
        raise ValueError(f"not a verdict: {raw!r}")
    error_type: Any = None
    error_code: Any = None
    if "error_type" in raw:
        error_type, error_code = raw["error_type"], raw.get("error_code")
        if not all(field is None or isinstance(field, str) for field in (error_type, error_code)):
            raise ValueError(f"not a verdict: {raw!r}")
    elif outcome == "error":
        error_type, error_code = _error_of(detail)
    return EngineVerdict(outcome, detail, error_type, error_code)


def _error_of(detail: str) -> tuple[str | None, str | None]:
    """The type and the code of the error that the detail of a failed run starts with.

    The head ends at the colon before the message, or at the end of a detail that has none.
    """
    found = _ERROR_HEAD.match(detail)
    return (found.group(1), found.group(2)) if found else (None, None)


# --- Output ---


def _render(
    cat: Catalogue, index: StatusIndex, meta: RunMeta, examples_root: Path
) -> dict[str, str]:
    """The text of every file by its path in the folder of the map, each diagram checked.

    ``index.rst`` is the entry page, with ``how_to_read.rst``, ``overview.rst``,
    ``operators.rst`` and ``decisions.rst`` beside it,
    ``<chapter>/index.rst`` the page of a block, and
    ``<chapter>/<operator>.rst`` the page of an operator, which reads the folders of its examples
    in ``examples_root``; ``run.html``, the versions that the heads of the pages read, then
    ``status.json`` and ``example_pages.json`` come after the pages. The
    diagrams of the pages are checked against the limits of Mermaid, and the error names the page.
    """
    files = {
        _INDEX_FILE: index_rst(cat, index, meta),
        _HOW_TO_READ_FILE: how_to_read_rst(),
        _OVERVIEW_FILE: overview_rst(cat, index, meta),
        _OPERATORS_FILE: operators_rst(cat, index),
        _DECISIONS_FILE: decisions_rst(cat),
    }
    for chapter in cat.chapters:
        files[f"{chapter.id}/{_INDEX_FILE}"] = block_rst(chapter, cat, index, meta)
        for op in chapter.operators:
            files[f"{chapter.id}/{op.id}.rst"] = operator_rst(
                chapter, op, cat, index, meta, examples_root=examples_root
            )
    for name, text in files.items():
        for block in mermaid_blocks(text):
            try:
                check_limits(block)
            except MermaidLimitError as exc:
                raise MermaidLimitError(f"{name}: {exc}") from exc
    files[RUN_FILE] = run_html(meta)
    files[_STATUS_FILE] = _status_json(meta, index)
    files[_EXAMPLE_PAGES_FILE] = _example_pages_json(cat, index)
    return files


def _status_json(meta: RunMeta, index: StatusIndex) -> str:
    """The results of every example: sorted keys, so that a rerun gives a clean diff.

    A verdict is written with all its fields: the outcome, the detail, and the type and the code
    of the error that the run raised (null where there is none).
    """
    examples = {
        key: {
            "pandas": asdict(result.pandas),
            "duckdb": asdict(result.duckdb),
            "engines_agree": result.engines_agree,
            "status": result.status,
        }
        for key, result in index.examples.items()
    }
    document = {"meta": asdict(meta), "examples": examples}
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _example_pages_json(cat: Catalogue, index: StatusIndex) -> str:
    """What the page of an example folder in the built site shows, for every example.

    It maps ``<chapter>/<operator>/<variant>/<example>`` to the title of the example, its status
    (``ok``, ``partial``, ``refused``, ``missing`` or ``diverge``), the mark of each engine as
    ``tool.status.mark`` gives it, the page of its operator relative to the folder of the map
    (``<chapter>/<operator>.html``) and the id of the example on that page, so that the page of
    the example can link back to it. The keys are sorted, as in ``status.json``.
    """
    pages = {}
    for chapter in cat.chapters:
        for op in chapter.operators:
            for variant in op.variants:
                for example in variant.examples:
                    key = f"{chapter.id}/{op.id}/{variant.id}/{example.name}"
                    result = index.examples[key]
                    pages[key] = {
                        "title": example_title(example),
                        "status": result.status,
                        "pandas": mark(result.pandas),
                        "duckdb": mark(result.duckdb),
                        "operator_page": f"{chapter.id}/{op.id}.html",
                        "anchor": example_anchor(variant, example),
                    }
    return json.dumps(pages, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# --- The manifest ---


def _read_manifest(root: Path) -> list[str]:
    """The paths that the previous run generated, from the manifest in ``root``; none if no file.

    A line that is not the path of a generated page raises ``GenerateError``: the manifest says
    what a run may remove, so a path outside the folder (``..``, an absolute path), in the
    examples, hidden, or of a file that people keep is not taken on trust. Nor is a path that goes
    through a symbolic link, whose file can be anywhere.
    """
    path = root / _MANIFEST
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as exc:
        raise GenerateError(
            f"{_MANIFEST}: cannot read the files of the previous run ({type(exc).__name__}: {exc})"
        ) from exc
    for number, line in enumerate(lines, 1):
        if not _is_generated_path(line):
            raise GenerateError(
                f"{_MANIFEST}: line {number} is not the path of a generated file: {line!r}; "
                f"fix the manifest, or remove it to start from an empty one"
            )
        if (root / line).resolve() != root.resolve() / line:
            raise GenerateError(
                f"{_MANIFEST}: line {number} goes through a symbolic link: {line!r}; "
                f"fix the manifest, or remove it to start from an empty one"
            )
    return lines


def _is_generated_path(text: str) -> bool:
    """Whether ``text`` can be the path of a file that a run generated, in the folder of the map.

    It is ``index.rst``, ``how_to_read.rst``, ``overview.rst``, ``operators.rst``,
    ``decisions.rst``, ``run.html`` (the versions that the heads of the pages read),
    ``status.json`` or
    ``example_pages.json`` in the folder of the map, or a page in the folder of a block:
    lower-case letters, digits and ``_`` (the rule of the ids of the catalogue), with ``.rst`` or
    ``.json``, and no folder that people keep in the map (the examples, the tool and the
    catalogue). So nothing else can match: no ``..``, no absolute or hidden path (the manifest, a
    staging folder), no source file of the tool.
    """
    return bool(_GENERATED_PATH.fullmatch(text)) and text.split("/")[0] not in _KEPT_FOLDERS


def _manifest_text(names: Iterable[str]) -> str:
    """The manifest of a run that generated ``names``: one path per line, sorted."""
    return "".join(f"{name}\n" for name in sorted(names))


# --- Writing ---


def _write_all(
    root: Path, files: Mapping[str, str], progress: _Progress | None = None
) -> list[str]:
    """Write the files to a folder of their own, then move each over the one in ``root``.

    The names are paths in ``root``, such as ``alpha/index.rst``: the folders that they need are
    made, in the folder of their own and in ``root``. The files that the manifest of the previous
    run lists and that ``files`` does not have are removed, and the manifest of this run is moved
    last (its path is not one of ``files``). Everything is written before anything is replaced or
    removed, so that a file that cannot be written leaves the old ones as they are. The folder is
    in ``root``: a move inside a filesystem replaces a file in one step. A move or a removal that
    fails puts back the files that were moved or removed before it, and removes the folders that
    were made. The folder of their own is removed at the end, unless an old file could not be put
    back: then it is kept, and the ``RestoreError`` says where it is. ``progress`` is told when
    every new file is in place, and while a staging folder is there to be removed. The result is
    the stale files that were removed, in order, as paths in ``root``.
    """
    progress = progress if progress is not None else _Progress()
    stale = sorted(
        name for name in set(_read_manifest(root)) - set(files) if (root / name).is_file()
    )
    staging = Path(tempfile.mkdtemp(prefix=".generate-", dir=root))
    progress.leftover = staging
    keep = False
    try:
        for name, text in files.items():
            _write_staged(staging / _NEW / name, text)
        _write_staged(staging / _NEW / _MANIFEST, _manifest_text(files))
        _replace_all(root, staging, list(files), stale)
        progress.written = True
        _remove_empty_folders(root, stale)
    except RestoreError:
        keep = True
        progress.leftover = None  # it is kept on purpose, and the error says where it is
        raise
    finally:
        if not keep:
            shutil.rmtree(staging, ignore_errors=True)
            if not staging.exists():
                progress.leftover = None
    return stale


def _write_staged(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` in the staging folder, making the folders of its name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _replace_all(root: Path, staging: Path, names: Sequence[str], stale: Sequence[str]) -> None:
    """Move ``names`` from ``staging`` over the files of ``root``, remove ``stale``, all or none.

    The files are moved first, then the stale files are removed, and the manifest is moved last:
    a run that is killed in between leaves the old manifest, which still lists every file that
    the next run is to remove. The old bytes of a file are copied aside before it is replaced or
    removed, and the file is noted before the move, so that a move that is cut short, even by an
    interrupt right after it, is undone too. If anything goes wrong, the files that were noted
    get their old bytes back, or are removed if they were new, the folders that the run made are
    removed if they are empty, and the error goes on. When a file cannot be put back, the others
    still are, and the ``RestoreError`` that follows names the ones that could not.
    """
    backups = staging / _OLD
    backups.mkdir()
    touched: list[str] = []
    made: list[Path] = []
    try:
        for name in names:
            _move(root, staging, name, touched, made)
        for name in stale:
            _back_up(root, backups, name)
            touched.append(name)
            (root / name).unlink(missing_ok=True)
        _move(root, staging, _MANIFEST, touched, made)
    except BaseException as exc:
        lost = _put_back(root, backups, touched)
        _remove_folders(made)
        if lost:
            cause = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            raise RestoreError(
                f"{cause}; could not put back {', '.join(lost)}; the staging folder {staging} "
                f"is kept, the old files are in its folder {backups.name}"
            ) from exc
        raise


def _move(root: Path, staging: Path, name: str, touched: list[str], made: list[Path]) -> None:
    """Move ``name`` from the new files of ``staging`` over the file of ``root``.

    The old bytes of the file are copied aside, and the file is noted in ``touched`` before the
    move. The folders of its name that ``root`` does not have are made first, and noted in
    ``made`` before each is made.
    """
    _back_up(root, staging / _OLD, name)
    touched.append(name)
    missing: list[Path] = []
    folder = (root / name).parent
    while folder != root and not folder.exists():
        missing.append(folder)
        folder = folder.parent
    for folder in reversed(missing):
        made.append(folder)
        folder.mkdir()
    os.replace(staging / _NEW / name, root / name)


def _back_up(root: Path, backups: Path, name: str) -> None:
    """Copy the file ``name`` of ``root``, if it is there, to the same name in ``backups``."""
    if (root / name).is_file():
        (backups / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, backups / name)


def _put_back(root: Path, backups: Path, names: Sequence[str]) -> list[str]:
    """Give ``names`` their old bytes in ``root``, or remove them if they are new.

    Every file is tried, whatever happened to the ones before it, a second interrupt included.
    The result is the ones that could not be put back, or are not known to be.
    """
    lost: list[str] = []
    for name in names:
        try:
            if (backups / name).is_file():
                os.replace(backups / name, root / name)
            else:
                (root / name).unlink(missing_ok=True)
        except BaseException:  # a failure, or a second interrupt, must not stop the others
            lost.append(name)
    return lost


def _remove_folders(folders: Sequence[Path]) -> None:
    """Remove the folders that a run made, the last one first, if they are empty.

    A folder that cannot be removed is left: it is empty or holds a file that could not be put
    back, and the others are still tried, a second interrupt included.
    """
    for folder in reversed(folders):
        with contextlib.suppress(BaseException):  # not empty, not there, or a second interrupt
            folder.rmdir()


def _remove_empty_folders(root: Path, names: Sequence[str]) -> None:
    """Remove the folders that removing ``names`` left empty, up to ``root`` and not it.

    The folder of a block that is gone is empty once its pages are removed, and so is the folder
    above it when it held nothing else. A folder that is not empty is kept, and so is every folder
    above it.
    """
    for name in names:
        folder = (root / name).parent
        while folder != root:
            try:
                folder.rmdir()
            except OSError:  # not empty, or already removed with the one before it
                break
            folder = folder.parent


if __name__ == "__main__":
    sys.exit(main())
