"""Write a Markdown summary of the test matrix, for the summary of a GitHub Actions run.

    python scripts/pytest_summary.py --title TITLE [--backends pandas,duckdb] FOLDER

``FOLDER`` has a folder per job of the matrix, as the jobs upload their results: a
``job.json`` with its ``os`` and ``python``, the JUnit XML file that pytest wrote for each
backend it ran (``pandas.xml``, ``duckdb.xml``), and ``coverage.txt`` with the total coverage,
when the job measured one, with ``coverage.json``, the report of ``coverage json``. The folders
of one OS and Python are one job of the summary. It has:

* a row per job and backend: its tests, passed, failed, errors, skipped, time and coverage; a
  backend without its file did not run (an earlier step failed, or the job was cancelled);
* the coverage of each folder of the package, from ``coverage.json``;
* the slowest tests, each once, with the job and the backend where it was slowest;
* the tests that failed or errored, with the first line of their message.

It only uses the standard library, so that it runs before anything is installed.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import xml.etree.ElementTree as ET  # noqa: S405  (the files are pytest's own, on the runner)
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

MESSAGE_MAX = 200
NONE = "—"


@dataclass
class Run:
    """A run of pytest: its counts, the time of each test, and the tests that failed."""

    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    time_s: float = 0.0
    times: dict[str, float] = field(default_factory=dict)  # test id -> seconds
    failed: list[tuple[str, str]] = field(default_factory=list)  # (test id, message)


@dataclass
class Job:
    """A job of the matrix: its OS and Python, its run of each backend, and its coverage."""

    os_name: str
    python: str
    runs: dict[str, Run | None]  # None: the file is there but cannot be read
    coverage: str | None = None

    def sort_key(self) -> tuple[str, tuple[int, ...]]:
        numbers = tuple(int(part) for part in self.python.split(".") if part.isdigit())
        return self.os_name, numbers


@dataclass
class FolderCoverage:
    """The statements of the files of a folder, and those that no test ran."""

    statements: int = 0
    missing: int = 0

    def percent(self) -> str:
        if not self.statements:
            return NONE
        return f"{100 * (self.statements - self.missing) / self.statements:.1f}%"


def folder_of(file_name: str) -> str:
    """
    The folder whose coverage a file counts in: the first level below the package, as
    ``vtlengine/AST`` for ``src/vtlengine/AST/Grammar/tokens.py``, or the package itself
    for the modules directly in it.
    """
    parts = file_name.replace("\\", "/").split("/")
    if "src" in parts:
        parts = parts[parts.index("src") + 1 :]
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0] if len(parts) > 1 else "."


def read_folder_coverage(folder: Path) -> dict[str, FolderCoverage]:
    """The coverage of each folder of the package, from the ``coverage.json`` of the jobs."""
    folders: dict[str, FolderCoverage] = {}
    for report in sorted(folder.glob("*/coverage.json")):
        files = json.loads(report.read_text(encoding="utf-8")).get("files", {})
        for file_name, data in files.items():
            coverage = folders.setdefault(folder_of(file_name), FolderCoverage())
            coverage.statements += int(data["summary"]["num_statements"])
            coverage.missing += int(data["summary"]["missing_lines"])
    return folders


def read_run(path: Path) -> Run:
    """The counts of every test suite of a JUnit XML file of pytest, its times and failures."""
    root = ET.parse(path).getroot()  # noqa: S314
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    run = Run()
    for suite in suites:
        run.tests += int(suite.get("tests", 0))
        run.failures += int(suite.get("failures", 0))
        run.errors += int(suite.get("errors", 0))
        run.skipped += int(suite.get("skipped", 0))
        run.time_s += float(suite.get("time", 0.0))
        for case in suite.iter("testcase"):
            test = f"{case.get('classname', '')}::{case.get('name', '')}"
            run.times[test] = max(run.times.get(test, 0.0), float(case.get("time", 0.0)))
            problem = case.find("failure")
            if problem is None:
                problem = case.find("error")
            if problem is not None:
                run.failed.append((test, problem.get("message", "")))
    return run


def _read_or_none(path: Path) -> Run | None:
    """The run of a JUnit XML file; None when it is empty or cut short, as by a crash."""
    try:
        return read_run(path)
    except ET.ParseError:
        return None


def read_jobs(folder: Path) -> list[Job]:
    """
    The jobs whose results are in the folders of ``folder``, in the order of the summary.

    The folders of one OS and Python make one job: a matrix that runs each backend in a job of
    its own, or measures the coverage in another, uploads a folder per job.
    """
    jobs: dict[tuple[str, str], Job] = {}
    for job_folder in sorted(path for path in folder.iterdir() if path.is_dir()):
        labels = job_folder / "job.json"
        if labels.is_file():
            data = json.loads(labels.read_text(encoding="utf-8"))
            os_name, python = str(data["os"]), str(data["python"])
        else:
            os_name, python = job_folder.name, NONE
        job = jobs.setdefault((os_name, python), Job(os_name, python, {}))
        job.runs.update(
            {path.stem: _read_or_none(path) for path in sorted(job_folder.glob("*.xml"))}
        )
        coverage_file = job_folder / "coverage.txt"
        if coverage_file.is_file():
            job.coverage = coverage_file.read_text(encoding="utf-8").strip() or None
    return sorted(jobs.values(), key=Job.sort_key)


def summarize(
    title: str,
    jobs: Sequence[Job],
    backends: Sequence[str] = ("pandas", "duckdb"),
    slowest: int = 5,
    limit: int = 30,
    folders: Mapping[str, FolderCoverage] | None = None,
) -> str:
    """
    The Markdown summary of ``jobs``: a row per job and backend, the coverage of each folder
    in ``folders``, the slowest tests and the failures.
    """
    lines = [
        f"### {title}",
        "",
        "| OS | Python | Backend | Tests | Passed | Failed | Errors | Skipped | Time | Coverage |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    slow: dict[str, tuple[float, str]] = {}  # test id -> (seconds, where)
    failed: list[tuple[str, str, str]] = []  # (where, test id, message)
    for job in jobs:
        for backend in backends:
            where = f"{job.os_name} | {job.python} | {backend}"
            coverage = f"{job.coverage}%" if job.coverage and backend == backends[-1] else NONE
            run = job.runs.get(backend)
            if run is None:
                state = "unreadable" if backend in job.runs else "did not run"
                dashes = " | ".join([NONE] * 5)
                lines.append(f"| {where} | {dashes} | {state} | {coverage} |")
                continue
            passed = run.tests - run.failures - run.errors - run.skipped
            lines.append(
                f"| {where} | {run.tests} | {passed} | {run.failures} | {run.errors} "
                f"| {run.skipped} | {duration(run.time_s)} | {coverage} |"
            )
            for test, seconds in run.times.items():
                if test not in slow or seconds > slow[test][0]:
                    slow[test] = (seconds, where)
            failed += [(where, test, message) for test, message in run.failed]
    if folders:
        total = FolderCoverage(
            sum(coverage.statements for coverage in folders.values()),
            sum(coverage.missing for coverage in folders.values()),
        )
        lines += [
            "",
            "#### 📊 Coverage by folder",
            "",
            "| Folder | Statements | Missed | Coverage |",
            "| --- | ---: | ---: | ---: |",
        ]
        lines += [
            f"| `{escaped(name)}` | {coverage.statements} | {coverage.missing} "
            f"| {coverage.percent()} |"
            for name, coverage in sorted(folders.items(), key=lambda item: item[0].lower())
        ]
        lines.append(f"| **Total** | {total.statements} | {total.missing} | {total.percent()} |")
    if slow:
        ranked = sorted(slow.items(), key=lambda item: -item[1][0])[:slowest]
        lines += [
            "",
            f"#### 🐢 The {len(ranked)} slowest tests",
            "",
            "| Test | Time | OS | Python | Backend |",
            "| --- | ---: | --- | --- | --- |",
        ]
        lines += [
            f"| `{escaped(test)}` | {seconds:.1f} s | {where} |"
            for test, (seconds, where) in ranked
        ]
    if failed:
        lines += [
            "",
            f"#### ❌ Failures and errors ({len(failed)})",
            "",
            "| OS | Python | Backend | Test | Message |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {where} | `{escaped(test)}` | {cell(message)} |"
            for where, test, message in failed[:limit]
        ]
        if len(failed) > limit:
            lines += ["", f"… and {len(failed) - limit} more"]
    return "\n".join(lines) + "\n"


def cell(text: str) -> str:
    """``text`` on one line, cut at ``MESSAGE_MAX`` characters, safe in a Markdown table."""
    one_line = " ".join(text.split())
    if len(one_line) > MESSAGE_MAX:
        one_line = one_line[: MESSAGE_MAX - 1] + "…"
    return escaped(html.escape(one_line, quote=False))


def escaped(text: str) -> str:
    """``text`` with its pipes escaped, which would split a cell of a Markdown table."""
    return text.replace("|", "\\|")


def duration(seconds: float) -> str:
    """``31s``, ``2m31s`` or ``1h02m``."""
    whole = int(seconds)
    if whole < 60:
        return f"{whole}s"
    if whole < 3600:
        return f"{whole // 60}m{whole % 60:02d}s"
    return f"{whole // 3600}h{whole % 3600 // 60:02d}m"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--title", required=True, help="the heading of the summary")
    parser.add_argument(
        "--backends", default="pandas,duckdb", help="the backends that every job runs, in order"
    )
    parser.add_argument("--slowest", type=int, default=5, help="how many slowest tests to list")
    parser.add_argument("--limit", type=int, default=30, help="the failures listed at most")
    parser.add_argument("folder", type=Path, help="a folder per job, with its results")
    args = parser.parse_args(argv)
    backends = [name for name in args.backends.split(",") if name]
    jobs = read_jobs(args.folder) if args.folder.is_dir() else []
    folders = read_folder_coverage(args.folder) if args.folder.is_dir() else {}
    sys.stdout.write(summarize(args.title, jobs, backends, args.slowest, args.limit, folders))
    return 0


if __name__ == "__main__":
    sys.exit(main())
