"""Generate the operator map: validate, run every example on both engines, judge, write.

    generate.py [--only CHAPTER]... [--workers 4] [--timeout 120] [--check-only]

It needs a Python that can import vtlengine, with the compiled parser in src/ ("Building the C++
parser from source" in CONTRIBUTING.md says how to build it): every example runs in a child
process that imports the engines of this worktree, and one child checks that it can before any
example runs. Where systemd is available, run it under a memory cap, from the repository root:

systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 python docs/operator_map/generate.py

Elsewhere the cap is optional.

1. The catalogue is checked against the index of the manual (``validate_catalogue``), and the
   example folders against the catalogue in both directions: a catalogue example needs a folder
   that loads, and a folder needs a catalogue example.
2. One child process imports vtlengine and both engines, and gives their versions (the
   preflight of ``tool.runner``). Then every example runs on the pandas and the DuckDB engine,
   each in a process of its own. A ``random`` example runs twice, to see whether its values are
   stable.
3. The status of each example comes from the two runs (``tool.status``), the statuses are rolled
   up, and ``index.rst``, one rst page per chapter and ``status.json`` are rendered. The pages are
   documents of the Sphinx docs of the repository (``docs/``), which copies the examples next to
   them when it builds the site.
4. Only when every file has been rendered, and every diagram is within the limits of Mermaid, are
   the files written: first to a folder of their own, then moved over the old ones.

With ``--only`` just the named chapters run (``types`` stands for the type hubs). The other
chapters keep their results from the ``status.json`` of the previous run: an example that is not
run and has no result there is a problem. That file is read only when a chapter with examples is
left out of ``--only``. The folders that are missing are checked only for the chapters named, and
the folders with no catalogue example always. Nothing is run or written with ``--check-only``: it
stops after the checks, and does not ask for previous results.

The exit code is 0 for a run that wrote its files, 1 for any problem (printed one per line, and
nothing is written), and 2 for a run that was interrupted, wherever the interrupt came, that hit
an internal error (one line: its type and message), or whose preflight child cannot import the
engines (one line: the reason; nothing is run or written). Until every new file is in place, a
failure or an interrupt leaves the files of the previous run as they were: the files that were
moved are put back. Only when putting one back fails too, even by a second interrupt, the folder
that has the old bytes is kept, and the line says where it is. After that point nothing can be
undone: an interrupt or an error in the clean-up or in the summary leaves the new files, and the
line says so, and names a ``.generate-*`` folder that the clean-up did not remove. This process
never imports vtlengine or pandas: the engines run in the children of ``tool.runner``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, get_args

from catalogue import manual_index
from tool.examples import ExampleError, ExampleFiles, example_dir, load_example, scan_example_dirs
from tool.mermaid import MermaidLimitError, check_limits
from tool.model import Catalogue, Example, Status, load_catalogue, validate_catalogue
from tool.render import RunMeta, chapter_rst, index_rst, mermaid_blocks
from tool.runner import Engine, EngineVersions, PreflightError, RunOutput, preflight, run_all
from tool.status import (
    EMOJI,
    EngineVerdict,
    ExampleResult,
    Outcome,
    StatusIndex,
    _status,
    build_index,
    counts,
    example_result,
)

ROOT = Path(__file__).resolve().parent  # docs/operator_map: where the files are written
REPO = ROOT.parents[1]  # the worktree: its git commit and vtlengine version are the engine's

_TYPES: Final = "types"  # the group of the examples of the type hubs, as a chapter of --only
_STATUS_FILE: Final = "status.json"
_MAX_UNRUN: Final = 10  # the "no result" problems that are printed, one line each
_ENGINES: Final[tuple[Engine, ...]] = ("pandas", "duckdb")
_VERSION: Final = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.MULTILINE)
# The start of the detail of a failed run, ``<type> <code>: <message>`` with the code optional.
# A ``status.json`` of the tool before the fields ``error_type`` and ``error_code`` says them so.
_ERROR_HEAD: Final = re.compile(r"^(\w+)(?: (\d+(?:-\d+)+))?:")
_INDEX_FILE: Final = "index.rst"
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

    group: str  # the chapter id, or ``types``
    operator: str  # the operator id, or the type id
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
    known = [*(chapter.id for chapter in cat.chapters), _TYPES]
    if any(name not in known for name in only):
        return _fail(
            [
                f"unknown chapter '{name}' for --only (known: {', '.join(known)})"
                for name in only
                if name not in known
            ]
        )

    items = _items(cat)
    checked = _check(cat, items, only, ROOT, with_previous=not args.check_only)
    if checked.problems:
        return _fail(checked.problems)
    todo = [item for item in items if not only or item.group in only]
    if args.check_only:
        print(f"ok: {len(todo)} examples checked, nothing was run")
        return 0
    try:
        versions = preflight(args.timeout)
    except PreflightError as exc:  # every run would crash: the pages must not take that
        print(str(exc), file=sys.stderr)
        return 2
    try:
        meta = _run_meta(versions)
    except GenerateError as exc:
        return _fail([str(exc)])

    examples_root = ROOT / "examples"
    print(f"running {len(todo)} examples on both engines ({args.workers} at a time)", flush=True)
    outputs = run_all(_jobs(todo, examples_root), args.workers, args.timeout)
    results = _judge(todo, checked.loaded, outputs, examples_root, meta.date)
    index = build_index(cat, {**checked.previous, **results})
    try:
        files = _render(cat, index, meta, examples_root)
    except MermaidLimitError as exc:
        return _fail([str(exc)])
    _write_all(ROOT, files, progress)
    summary = " · ".join(
        f"{EMOJI[status]} {number}"
        for status, number in counts([each.status for each in index.examples.values()]).items()
    )
    print(f"wrote {', '.join(sorted(files))} ({summary} examples)")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate the VTL operator map: run every example on both engines, "
        "judge the results and write the Sphinx pages (rst) and status.json.",
        epilog=_HOW_TO_RUN,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="CHAPTER",
        help="run only this chapter, and keep the results of the others from the previous run; "
        "'types' is the type hubs; repeat it to name several chapters",
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
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="check the catalogue and the example folders, then stop: nothing is run or written",
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
    loaded: dict[str, ExampleFiles]  # the examples that are to run, by key
    previous: dict[str, ExampleResult]  # the results of the previous run, with ``--only``


def _check(
    cat: Catalogue, items: Sequence[_Item], only: Sequence[str], root: Path, *, with_previous: bool
) -> _Checked:
    """Check the catalogue and the example folders, and read the previous results if wanted.

    The previous results are read only when ``only`` leaves out a chapter that has examples, and
    an example that is not run and has none is a problem. They are not needed to check, so
    ``with_previous`` leaves them out.
    """
    problems = _catalogue_problems(cat, only)
    folder_problems, loaded = _folder_problems(items, only, root)
    problems.extend(folder_problems)
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
    """Every example of the catalogue, in its order: chapters, then the types."""
    items = [
        _Item(chapter.id, operator.id, variant.id, example)
        for chapter in cat.chapters
        for operator in chapter.operators
        for variant in operator.variants
        for example in variant.examples
    ]
    items.extend(
        _Item(_TYPES, typedef.id, variant.id, example)
        for typedef in cat.types
        for variant in typedef.variants
        for example in variant.examples
    )
    return items


def _catalogue_problems(cat: Catalogue, only: Sequence[str]) -> list[str]:
    """The problems of the catalogue against the manual, and what cannot be rolled up."""
    problems = validate_catalogue(
        cat, manual_index.MANUAL_PAGES, manual_index.MANUAL_EXAMPLES, scope=only or None
    )
    for chapter in cat.chapters:
        if not chapter.operators:
            problems.append(f"{chapter.id}: no operator")
        for operator in chapter.operators:
            if not operator.variants:
                problems.append(f"{chapter.id}.{operator.id}: no variant")
            problems.extend(
                f"{chapter.id}.{operator.id}.{variant.id}: no example"
                for variant in operator.variants
                if not variant.examples
            )
    for typedef in cat.types:
        problems.extend(
            f"types.{typedef.id}.{variant.id}: no example"
            for variant in typedef.variants
            if not variant.examples
        )
    return list(dict.fromkeys(problems))


def _folder_problems(
    items: Sequence[_Item], only: Sequence[str], root: Path
) -> tuple[list[str], dict[str, ExampleFiles]]:
    """The folders of the examples against the catalogue, and the examples that were loaded.

    A catalogue example needs a folder that loads, but only for the chapters named by ``only``
    (all of them without it). A folder with no catalogue example is a problem whatever ``only``
    says.
    """
    examples_root = root / "examples"
    on_disk = scan_example_dirs(examples_root)
    problems: list[str] = []
    loaded: dict[str, ExampleFiles] = {}
    for item in items:
        if only and item.group not in only:
            continue
        path = example_dir(examples_root, *item.parts)
        if not path.is_dir():
            problems.append(f"missing example folder: {path.relative_to(root).as_posix()}")
            continue
        try:
            loaded[item.key] = load_example(path)
        except ExampleError as exc:
            problems.append(str(exc))
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
    return ExampleResult(pandas, duckdb, agree, _status(pandas, duckdb, agree))


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
    """The type and the code of the error that the detail of a failed run starts with."""
    found = _ERROR_HEAD.match(detail)
    return (found.group(1), found.group(2)) if found else (None, None)


# --- Output ---


def _render(
    cat: Catalogue, index: StatusIndex, meta: RunMeta, examples_root: Path
) -> dict[str, str]:
    """The text of every file, with each diagram checked against the limits of Mermaid."""
    files = {_INDEX_FILE: index_rst(cat, index, meta, examples_root=examples_root)}
    for chapter in cat.chapters:
        files[f"{chapter.id}.rst"] = chapter_rst(
            chapter, cat, index, meta, examples_root=examples_root
        )
    for name, text in files.items():
        for block in mermaid_blocks(text):
            try:
                check_limits(block)
            except MermaidLimitError as exc:
                raise MermaidLimitError(f"{name}: {exc}") from exc
    files[_STATUS_FILE] = _status_json(meta, index)
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


def _write_all(root: Path, files: Mapping[str, str], progress: _Progress | None = None) -> None:
    """Write the files to a folder of their own, then move each over the one in ``root``.

    Everything is written before anything is replaced, so that a file that cannot be written
    leaves the old ones as they are. The folder is in ``root``: a move inside a filesystem
    replaces a file in one step. A move that fails puts back the files that were moved before it.
    The folder is removed at the end, unless an old file could not be put back: then it is kept,
    and the ``RestoreError`` says where it is. ``progress`` is told when every new file is in
    place, and while a staging folder is there to be removed.
    """
    progress = progress if progress is not None else _Progress()
    staging = Path(tempfile.mkdtemp(prefix=".generate-", dir=root))
    progress.leftover = staging
    keep = False
    try:
        for name, text in files.items():
            with (staging / name).open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        _replace_all(root, staging, list(files))
        progress.written = True
    except RestoreError:
        keep = True
        progress.leftover = None  # it is kept on purpose, and the error says where it is
        raise
    finally:
        if not keep:
            shutil.rmtree(staging, ignore_errors=True)
            if not staging.exists():
                progress.leftover = None


def _replace_all(root: Path, staging: Path, names: Sequence[str]) -> None:
    """Move ``names`` from ``staging`` over the files of ``root``, all or none.

    The old bytes of a file are copied aside before it is replaced, and the file is noted before
    the move, so that a move that is cut short, even by an interrupt right after it, is undone
    too. If anything goes wrong, the files that were noted get their old bytes back, or are
    removed if they were new, and the error goes on. When a file cannot be put back, the others
    still are, and the ``RestoreError`` that follows names the ones that could not.
    """
    backups = staging / "old"
    backups.mkdir()
    replaced: list[str] = []
    try:
        for name in names:
            if (root / name).is_file():
                shutil.copy2(root / name, backups / name)
            replaced.append(name)
            os.replace(staging / name, root / name)
    except BaseException as exc:
        lost = _put_back(root, backups, replaced)
        if lost:
            cause = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            raise RestoreError(
                f"{cause}; could not put back {', '.join(lost)}; the staging folder {staging} "
                f"is kept, the old files are in its folder {backups.name}"
            ) from exc
        raise


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


if __name__ == "__main__":
    sys.exit(main())
