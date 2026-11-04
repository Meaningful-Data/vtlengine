import json
import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import the module
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from pytest_summary import main, read_folder_coverage, read_jobs, summarize


def junit(*cases: str, tests: int, failures: int = 0, errors: int = 0, skipped: int = 0) -> str:
    """A JUnit XML file of pytest with ``cases`` and the counts of its test suite."""
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" errors="{errors}" failures="{failures}" skipped="{skipped}" '
        f'tests="{tests}" time="151.25">{"".join(cases)}</testsuite></testsuites>'
    )


def case(test: str, time_s: float, problem: str = "") -> str:
    module, name = test.split("::")
    return f'<testcase classname="{module}" name="{name}" time="{time_s}">{problem}</testcase>'


def job(root: Path, os_name: str, python: str, coverage: str | None = "91", **runs: str) -> Path:
    """The results of a job of the matrix, as the job uploads them: a folder per job."""
    folder = root / f"test-results-{os_name}-{python}"
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(json.dumps({"os": os_name, "python": python}))
    if coverage is not None:
        (folder / "coverage.txt").write_text(f"{coverage}\n")
    for backend, text in runs.items():
        (folder / f"{backend}.xml").write_text(text, encoding="utf-8")
    return folder


def rows(text: str) -> list[str]:
    return text.splitlines()


def test_each_job_and_backend_is_a_row(tmp_path: Path) -> None:
    passing = junit(case("tests.A.test_a::test_x", 0.1), tests=10, skipped=2)
    failing = junit(
        case("tests.B.test_b::test_y", 0.2, '<failure message="no"/>'), tests=12, failures=2
    )
    job(tmp_path, "windows-latest", "3.10", pandas=passing, duckdb=failing)
    job(tmp_path, "ubuntu-latest", "3.13", pandas=passing, duckdb=passing)
    text = summarize("Tests", read_jobs(tmp_path))
    lines = rows(text)
    assert lines[0] == "### Tests"
    table = [
        line for line in lines if line.startswith(("| ubuntu", "| windows")) and "`" not in line
    ]
    assert table == [
        "| ubuntu-latest | 3.13 | pandas | 10 | 8 | 0 | 0 | 2 | 2m31s | — |",
        "| ubuntu-latest | 3.13 | duckdb | 10 | 8 | 0 | 0 | 2 | 2m31s | 91% |",
        "| windows-latest | 3.10 | pandas | 10 | 8 | 0 | 0 | 2 | 2m31s | — |",
        "| windows-latest | 3.10 | duckdb | 12 | 10 | 2 | 0 | 0 | 2m31s | 91% |",
    ]


def test_the_slowest_tests_are_listed_once_where_they_were_slowest(tmp_path: Path) -> None:
    times = {f"tests.T.test_t::test_{n}": float(n) for n in range(1, 8)}
    fast = junit(*(case(test, time_s) for test, time_s in times.items()), tests=7)
    # On Windows, test_7 takes longer than anywhere else.
    slow = junit(case("tests.T.test_t::test_7", 30.5), tests=1)
    job(tmp_path, "ubuntu-latest", "3.13", pandas=fast, duckdb=fast)
    job(tmp_path, "windows-latest", "3.11", pandas=fast, duckdb=slow)
    text = summarize("Tests", read_jobs(tmp_path))
    assert "#### 🐢 The 5 slowest tests" in text
    slowest = [line for line in rows(text) if line.startswith("| `tests.T")]
    assert slowest == [
        "| `tests.T.test_t::test_7` | 30.5 s | windows-latest | 3.11 | duckdb |",
        "| `tests.T.test_t::test_6` | 6.0 s | ubuntu-latest | 3.13 | pandas |",
        "| `tests.T.test_t::test_5` | 5.0 s | ubuntu-latest | 3.13 | pandas |",
        "| `tests.T.test_t::test_4` | 4.0 s | ubuntu-latest | 3.13 | pandas |",
        "| `tests.T.test_t::test_3` | 3.0 s | ubuntu-latest | 3.13 | pandas |",
    ]


def test_failures_and_errors_name_the_job_and_the_backend(tmp_path: Path) -> None:
    failing = junit(
        case(
            "tests.Bugs.test_bugs.BugsTests::test_1",
            0.2,
            '<failure message="AssertionError: a | b&#10;second line"/>',
        ),
        case("tests.API.test_api::test_3", 0.2, "<error message=\"KeyError: 'x'\"/>"),
        tests=12,
        failures=1,
        errors=1,
    )
    job(tmp_path, "macos-latest", "3.12", duckdb=failing)
    text = summarize("Tests", read_jobs(tmp_path))
    assert "#### ❌ Failures and errors (2)" in text
    assert (
        "| macos-latest | 3.12 | duckdb | `tests.Bugs.test_bugs.BugsTests::test_1` "
        "| AssertionError: a \\| b second line |"
    ) in rows(text)
    assert "| macos-latest | 3.12 | duckdb | `tests.API.test_api::test_3` | KeyError: 'x' |" in (
        rows(text)
    )


def test_the_folders_of_one_os_and_python_are_one_job(tmp_path: Path) -> None:
    """A job per backend, and the coverage measured in a job of its own."""
    labels = json.dumps({"os": "ubuntu-latest", "python": "3.12"})
    for backend in ("pandas", "duckdb"):
        folder = tmp_path / f"test-results-ubuntu-latest-3.12-{backend}"
        folder.mkdir()
        (folder / "job.json").write_text(labels)
        (folder / f"{backend}.xml").write_text(junit(tests=10, skipped=2), encoding="utf-8")
    coverage = tmp_path / "test-results-coverage"
    coverage.mkdir()
    (coverage / "job.json").write_text(labels)
    (coverage / "coverage.txt").write_text("94\n")
    lines = rows(summarize("Tests", read_jobs(tmp_path)))
    assert [line for line in lines if line.startswith("| ubuntu-latest")] == [
        "| ubuntu-latest | 3.12 | pandas | 10 | 8 | 0 | 0 | 2 | 2m31s | — |",
        "| ubuntu-latest | 3.12 | duckdb | 10 | 8 | 0 | 0 | 2 | 2m31s | 94% |",
    ]


def coverage_report(*files: tuple[str, int, int]) -> str:
    """A ``coverage json`` report with ``files`` as (name, statements, missed lines)."""
    return json.dumps(
        {
            "files": {
                name: {"summary": {"num_statements": statements, "missing_lines": missing}}
                for name, statements, missing in files
            }
        }
    )


def test_the_coverage_of_each_folder_of_the_package_is_a_row(tmp_path: Path) -> None:
    folder = job(tmp_path, "ubuntu-latest", "3.12", coverage="94")
    (folder / "coverage.json").write_text(
        coverage_report(
            ("src/vtlengine/__init__.py", 20, 0),
            ("src/vtlengine/API/__init__.py", 300, 30),
            ("src/vtlengine/API/_InternalApi.py", 100, 10),
            ("src/vtlengine/AST/Grammar/tokens.py", 80, 0),
            ("src/vtlengine/Utils/_duckdb.py", 0, 0),
        )
    )
    jobs, folders = read_jobs(tmp_path), read_folder_coverage(tmp_path)
    text = summarize("Tests", jobs, folders=folders)
    assert "#### 📊 Coverage by folder" in text
    table = [line for line in rows(text) if line.startswith(("| `vtlengine", "| **Total"))]
    assert table == [
        "| `vtlengine` | 20 | 0 | 100.0% |",
        "| `vtlengine/API` | 400 | 40 | 90.0% |",
        "| `vtlengine/AST` | 80 | 0 | 100.0% |",
        "| `vtlengine/Utils` | 0 | 0 | — |",
        "| **Total** | 500 | 40 | 92.0% |",
    ]


def test_without_a_coverage_report_there_is_no_folder_table(tmp_path: Path) -> None:
    job(tmp_path, "ubuntu-latest", "3.12", pandas=junit(tests=1))
    folders = read_folder_coverage(tmp_path)
    assert folders == {}
    assert "Coverage by folder" not in summarize("Tests", read_jobs(tmp_path), folders=folders)


def test_main_writes_the_coverage_of_each_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = job(tmp_path, "ubuntu-latest", "3.12", duckdb=junit(tests=1))
    (folder / "coverage.json").write_text(coverage_report(("src/vtlengine/Model/x.py", 10, 1)))
    assert main(["--title", "Tests", str(tmp_path)]) == 0
    assert "| `vtlengine/Model` | 10 | 1 | 90.0% |" in rows(capsys.readouterr().out)


def test_a_backend_without_its_file_did_not_run(tmp_path: Path) -> None:
    job(tmp_path, "ubuntu-latest", "3.10", coverage=None, pandas=junit(tests=0))
    lines = rows(summarize("Tests", read_jobs(tmp_path)))
    assert "| ubuntu-latest | 3.10 | duckdb | — | — | — | — | — | did not run | — |" in lines


def test_a_passing_matrix_has_no_failure_section(tmp_path: Path) -> None:
    job(tmp_path, "ubuntu-latest", "3.10", pandas=junit(tests=1), duckdb=junit(tests=1))
    assert "Failures" not in summarize("Tests", read_jobs(tmp_path))


def test_the_failures_stop_at_the_limit(tmp_path: Path) -> None:
    many = junit(
        *(case(f"tests.t::test_{n}", 0.1, '<failure message="no"/>') for n in range(45)),
        tests=45,
        failures=45,
    )
    job(tmp_path, "ubuntu-latest", "3.10", pandas=many)
    text = summarize("Tests", read_jobs(tmp_path), limit=30)
    assert (
        sum(1 for line in rows(text) if line.startswith("| ubuntu-latest | 3.10 | pandas | `"))
        == 30
    )
    assert "… and 15 more" in text


def test_main_writes_the_summary_of_a_folder_of_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job(tmp_path, "ubuntu-latest", "3.13", pandas=junit(tests=10, skipped=2))
    assert main(["--title", "Tests", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "| ubuntu-latest | 3.13 | pandas | 10 | 8 | 0 | 0 | 2 | 2m31s | — |" in rows(out)


def test_a_test_id_with_a_pipe_stays_in_its_cell(tmp_path: Path) -> None:
    run = junit(case("tests.t::test_x[a|b]", 9.0, '<failure message="no"/>'), tests=1, failures=1)
    job(tmp_path, "ubuntu-latest", "3.13", pandas=run)
    lines = rows(summarize("Tests", read_jobs(tmp_path)))
    assert "| `tests.t::test_x[a\\|b]` | 9.0 s | ubuntu-latest | 3.13 | pandas |" in lines
    assert "| ubuntu-latest | 3.13 | pandas | `tests.t::test_x[a\\|b]` | no |" in lines


def test_an_unreadable_file_is_a_row_of_its_own(tmp_path: Path) -> None:
    job(
        tmp_path, "ubuntu-latest", "3.13", coverage=None, pandas="<testsuites><testsuite", duckdb=""
    )
    lines = rows(summarize("Tests", read_jobs(tmp_path)))
    assert "| ubuntu-latest | 3.13 | pandas | — | — | — | — | — | unreadable | — |" in lines
    assert "| ubuntu-latest | 3.13 | duckdb | — | — | — | — | — | unreadable | — |" in lines


def test_angle_brackets_of_a_message_are_shown(tmp_path: Path) -> None:
    message = "assert &lt;NA&gt; == 1 for &lt;vtlengine.Model.Dataset object&gt;"
    run = junit(
        case("tests.t::test_x", 1.0, f'<failure message="{message}"/>'), tests=1, failures=1
    )
    job(tmp_path, "ubuntu-latest", "3.13", pandas=run)
    text = summarize("Tests", read_jobs(tmp_path))
    assert "assert &lt;NA&gt; == 1 for &lt;vtlengine.Model.Dataset object&gt;" in text
