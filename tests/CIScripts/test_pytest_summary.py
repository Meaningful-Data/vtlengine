import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import the module
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from pytest_summary import main, summarize

PASSING = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0" skipped="2"
 tests="10" time="151.25"><testcase classname="tests.A.test_a" name="test_x" time="0.1"/>
</testsuite></testsuites>
"""

FAILING = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="1" failures="2" skipped="0"
 tests="12" time="62.0">
<testcase classname="tests.Bugs.test_bugs.BugsTests" name="test_1" time="0.2">
<failure message="AssertionError: a | b&#10;second line">trace</failure></testcase>
<testcase classname="tests.Bugs.test_bugs.BugsTests" name="test_2" time="0.2">
<failure message="AssertionError: 1 != 2">trace</failure></testcase>
<testcase classname="tests.API.test_api" name="test_3" time="0.2">
<error message="failed on setup with &quot;KeyError: 'x'&quot;">trace</error></testcase>
</testsuite></testsuites>
"""


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_each_run_is_a_row_with_its_counts_and_time(tmp_path: Path) -> None:
    pandas = write(tmp_path / "pandas.xml", PASSING)
    duckdb = write(tmp_path / "duckdb.xml", FAILING)
    text = summarize(
        "Tests · ubuntu-latest · Python 3.13", [("pandas", pandas), ("duckdb", duckdb)]
    )
    lines = text.splitlines()
    assert lines[0] == "### Tests · ubuntu-latest · Python 3.13"
    assert "| pandas | 10 | 8 | 0 | 0 | 2 | 2m31s |" in lines
    assert "| duckdb | 12 | 9 | 2 | 1 | 0 | 1m02s |" in lines


def test_failures_and_errors_are_listed_with_their_message(tmp_path: Path) -> None:
    duckdb = write(tmp_path / "duckdb.xml", FAILING)
    text = summarize("Tests", [("duckdb", duckdb)])
    assert "#### ❌ Failures and errors (3)" in text
    assert (
        "| duckdb | `tests.Bugs.test_bugs.BugsTests::test_1` | AssertionError: a \\| b second line |"
        in text.splitlines()
    )
    assert "| duckdb | `tests.API.test_api::test_3` | failed on setup with \"KeyError: 'x'\" |" in (
        text.splitlines()
    )


def test_a_run_without_its_file_did_not_run(tmp_path: Path) -> None:
    text = summarize("Tests", [("pandas", tmp_path / "missing.xml")])
    assert "| pandas | — | — | — | — | — | did not run |" in text.splitlines()


def test_a_passing_job_has_no_failure_section(tmp_path: Path) -> None:
    text = summarize("Tests", [("pandas", write(tmp_path / "pandas.xml", PASSING))])
    assert "Failures" not in text


def test_the_failures_stop_at_the_limit(tmp_path: Path) -> None:
    cases = "".join(
        f'<testcase classname="tests.t" name="test_{n}"><failure message="no"/></testcase>'
        for n in range(45)
    )
    many = (
        '<testsuites><testsuite errors="0" failures="45" skipped="0" tests="45" time="1">'
        f"{cases}</testsuite></testsuites>"
    )
    text = summarize("Tests", [("pandas", write(tmp_path / "pandas.xml", many))], limit=30)
    assert sum(1 for line in text.splitlines() if line.startswith("| pandas | `")) == 30
    assert "… and 15 more" in text


def test_main_writes_the_summary_of_name_and_file_pairs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pandas = write(tmp_path / "pandas.xml", PASSING)
    assert main(["--title", "Tests", f"pandas={pandas}"]) == 0
    assert "| pandas | 10 | 8 | 0 | 0 | 2 | 2m31s |" in capsys.readouterr().out.splitlines()
