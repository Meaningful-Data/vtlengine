import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

import pytest

# Add docs/scripts to path so we can import the modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "scripts"))

import build_multiversion as bm

SO_NAME = "vtl_cpp_parser.cpython-313-x86_64-linux-gnu.so"


class FakeRun:
    """Stands for ``subprocess.run``: records each command and answers as told."""

    def __init__(self, generator_code: int = 0, timeout: bool = False) -> None:
        self.commands: list[list[str]] = []
        self.generator_code = generator_code
        self.timeout = timeout

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append([str(part) for part in command])
        if command[1:3] == ["-m", "pip"]:  # pip download: a wheel with the parser
            dest = Path(command[command.index("--dest") + 1])
            version = command[command.index("download") + 1].split("==")[1]
            with zipfile.ZipFile(dest / f"vtlengine-{version}-cp313-linux.whl", "w") as wheel:
                wheel.writestr(f"vtlengine/AST/Grammar/_cpp_parser/{SO_NAME}", b"so")
            return subprocess.CompletedProcess(command, 0, "", "")
        if self.timeout:
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 0))
        return subprocess.CompletedProcess(command, self.generator_code, "", "")

    def generator_runs(self) -> list[list[str]]:
        return [command for command in self.commands if command[1].endswith("generate.py")]


def make_worktree(tmp_path: Path, with_generator: bool = True) -> Path:
    worktree = tmp_path / "wt"
    (worktree / bm.PARSER_DIR).mkdir(parents=True)
    if with_generator:
        (worktree / bm.MAP_GENERATOR).parent.mkdir(parents=True)
        (worktree / bm.MAP_GENERATOR).write_text("# the generator\n", encoding="utf-8")
    return worktree


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A checkout whose src/ has the compiled parser, as docs.yml leaves it."""
    root = tmp_path / "checkout"
    (root / bm.PARSER_DIR).mkdir(parents=True)
    (root / bm.PARSER_DIR / SO_NAME).write_bytes(b"checkout so")
    monkeypatch.setattr(bm, "REPO_ROOT", root)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setenv("VTLENGINE_DOCS_REGENERATE_MAP", "1")
    return root


def same_sources(monkeypatch: pytest.MonkeyPatch, same: bool) -> None:
    monkeypatch.setattr(bm, "_same_parser_sources", lambda worktree: same)


@pytest.mark.parametrize(
    ("ref", "version"),
    [
        ("v1.9.7", "1.9.7"),
        ("v1.9.7rc1", "1.9.7rc1"),
        ("v2.0.0rc3", "2.0.0rc3"),
        ("main", None),
        ("cr-1242", None),
        ("1.9.X", None),
    ],
)
def test_wheel_version_of_tags_and_branches(ref: str, version: Optional[str]) -> None:
    assert bm.wheel_version(ref) == version


def test_versions_without_the_generator_are_left_alone(
    tmp_path: Path, checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = FakeRun()
    monkeypatch.setattr(subprocess, "run", run)
    bm.regenerate_operator_map(make_worktree(tmp_path, with_generator=False), "v1.8.0")
    assert run.commands == []


def test_parser_comes_from_the_checkout_when_its_sources_match(
    tmp_path: Path, checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same_sources(monkeypatch, True)
    worktree = make_worktree(tmp_path)
    assert bm.install_parser(worktree, "v1.9.7") == "the checkout's build"
    assert (worktree / bm.PARSER_DIR / SO_NAME).read_bytes() == b"checkout so"


def test_parser_comes_from_pypi_otherwise(
    tmp_path: Path, checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same_sources(monkeypatch, False)
    run = FakeRun()
    monkeypatch.setattr(subprocess, "run", run)
    worktree = make_worktree(tmp_path)
    assert bm.install_parser(worktree, "v1.9.7") == "vtlengine==1.9.7 from PyPI"
    assert (worktree / bm.PARSER_DIR / SO_NAME).read_bytes() == b"so"
    [download] = run.commands
    assert download[:5] == [sys.executable, "-m", "pip", "download", "vtlengine==1.9.7"]


def test_a_branch_without_a_release_is_built_from_its_committed_pages(
    tmp_path: Path,
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    same_sources(monkeypatch, False)
    run = FakeRun()
    monkeypatch.setattr(subprocess, "run", run)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    bm.regenerate_operator_map(make_worktree(tmp_path), "main")
    assert run.commands == []
    out = capsys.readouterr().out
    assert "operator map of main: not regenerated (main has no release on PyPI" in out
    assert out.rstrip().endswith("built from its committed pages")
    assert summary.read_text(encoding="utf-8").strip() == out.strip()


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_a_failed_run_never_raises_and_is_reported(
    tmp_path: Path, checkout: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    same_sources(monkeypatch, True)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(subprocess, "run", FakeRun(generator_code=1, timeout=failure == "timeout"))
    bm.regenerate_operator_map(make_worktree(tmp_path), "v1.9.7")
    text = summary.read_text(encoding="utf-8")
    reason = "the generator exited with 1" if failure == "exit" else "timed out"
    assert "operator map of v1.9.7: not regenerated (" in text
    assert reason in text
    assert "built from its committed pages" in text


def test_summary_only_when_the_variable_is_set(
    tmp_path: Path, checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same_sources(monkeypatch, True)
    run = FakeRun()
    monkeypatch.setattr(subprocess, "run", run)
    worktree = make_worktree(tmp_path)
    bm.regenerate_operator_map(worktree, "v1.9.7")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    bm.regenerate_operator_map(worktree, "v1.9.7")
    without, with_summary = run.generator_runs()
    assert without == [sys.executable, str(worktree / bm.MAP_GENERATOR)]
    assert with_summary == [*without, "--summary", str(summary)]
    assert not summary.exists()  # the generator writes it; a run that went well adds nothing


def test_maps_are_regenerated_only_when_asked(
    tmp_path: Path, checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same_sources(monkeypatch, True)
    run = FakeRun()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.delenv("VTLENGINE_DOCS_REGENERATE_MAP")
    bm.regenerate_operator_map(make_worktree(tmp_path), "v1.9.7")
    assert run.commands == []  # a local build keeps the committed pages, and its minutes


def test_a_tag_whose_wheel_cannot_be_downloaded_is_built_from_its_committed_pages(
    tmp_path: Path, checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same_sources(monkeypatch, False)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    def no_wheel(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        error = "ERROR: No matching distribution found for vtlengine==9.9.9"
        return subprocess.CompletedProcess(command, 1, "", f"Looking in indexes\n{error}\n")

    monkeypatch.setattr(subprocess, "run", no_wheel)
    bm.regenerate_operator_map(make_worktree(tmp_path), "v9.9.9")
    assert summary.read_text(encoding="utf-8") == (
        "\noperator map of v9.9.9: not regenerated (pip cannot download vtlengine==9.9.9: "
        "ERROR: No matching distribution found for vtlengine==9.9.9); built from its committed "
        "pages\n"
    )
