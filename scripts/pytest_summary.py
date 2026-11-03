"""Write a Markdown summary of pytest runs, for the summary of a GitHub Actions job.

    python scripts/pytest_summary.py --title TITLE NAME=JUNIT_XML [NAME=JUNIT_XML ...]

Each ``NAME=JUNIT_XML`` is a run of pytest and the JUnit XML file that ``--junitxml`` wrote for
it. The summary has a row per run (its tests, passed, failed, errors, skipped and time), then
the tests that failed or errored, with the first line of their message, at most ``--limit`` of
them. A run whose file is not there did not run (an earlier step failed). It only uses the
standard library, so that it runs on every Python of the matrix before anything is installed.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET  # noqa: S405  (the files are pytest's own, on the runner)
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

MESSAGE_MAX = 200
NONE = "—"


@dataclass
class Run:
    """The counts of a run of pytest, and its tests that failed or errored."""

    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    time_s: float = 0.0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (test id, message)


def read_run(path: Path) -> Run:
    """The counts of every test suite of a JUnit XML file of pytest, and its failed tests."""
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
            problem = case.find("failure")
            if problem is None:
                problem = case.find("error")
            if problem is not None:
                test = f"{case.get('classname', '')}::{case.get('name', '')}"
                run.failed.append((test, problem.get("message", "")))
    return run


def summarize(title: str, runs: Sequence[tuple[str, Path]], limit: int = 30) -> str:
    """The Markdown summary of ``runs``, each a name and the JUnit XML file of its run."""
    lines = [
        f"### {title}",
        "",
        "| Run | Tests | Passed | Failed | Errors | Skipped | Time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    failed: list[tuple[str, str, str]] = []
    for name, path in runs:
        if not path.is_file():
            lines.append(f"| {name} | {NONE} | {NONE} | {NONE} | {NONE} | {NONE} | did not run |")
            continue
        run = read_run(path)
        passed = run.tests - run.failures - run.errors - run.skipped
        lines.append(
            f"| {name} | {run.tests} | {passed} | {run.failures} | {run.errors} | {run.skipped} "
            f"| {duration(run.time_s)} |"
        )
        failed += [(name, test, message) for test, message in run.failed]
    if failed:
        lines += [
            "",
            f"#### ❌ Failures and errors ({len(failed)})",
            "",
            "| Run | Test | Message |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| {name} | `{test}` | {cell(message)} |" for name, test, message in failed[:limit]
        ]
        if len(failed) > limit:
            lines += ["", f"… and {len(failed) - limit} more"]
    return "\n".join(lines) + "\n"


def cell(text: str) -> str:
    """``text`` on one line, cut at ``MESSAGE_MAX`` characters, safe in a Markdown table."""
    one_line = " ".join(text.split())
    if len(one_line) > MESSAGE_MAX:
        one_line = one_line[: MESSAGE_MAX - 1] + "…"
    return one_line.replace("|", "\\|")


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
    parser.add_argument("--limit", type=int, default=30, help="the failures listed at most")
    parser.add_argument("runs", nargs="+", metavar="NAME=JUNIT_XML", help="a run and its file")
    args = parser.parse_args(argv)
    runs = []
    for item in args.runs:
        name, _, path = item.partition("=")
        if not path:
            parser.error(f"not NAME=JUNIT_XML: {item}")
        runs.append((name, Path(path)))
    sys.stdout.write(summarize(args.title, runs, args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
