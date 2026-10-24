# ruff: noqa: S101
"""Checks for the generator: validation, the jobs it starts, merging, and the files it writes.

The runner is replaced by canned outputs, the catalogue and the manual index by small hand-made
ones, and the folder that the generator writes to by a temporary one: no engine runs here. Each
check gets a map of two chapters and a type, with their example folders on disk.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import generate
import pytest
from catalogue import manual_index

from tool.compare import ActualComponent, ActualDataset, ActualScalar
from tool.mermaid import MermaidLimitError
from tool.model import Catalogue, Chapter, Example, Operator, TypeDef, Variant, authored, manual
from tool.runner import Engine, RunOutput

TODAY = "2026-10-23"
SPEC_COMMIT = "140e946ed3f888951adb8bb66ea45030be84e2aa"
COMPONENTS = {
    "Id_1": ActualComponent("Identifier", "Integer", False),
    "Me_1": ActualComponent("Measure", "Number", True),
}
RIGHT_ROWS: list[dict[str, str | None]] = [
    {"Id_1": "1", "Me_1": "10.5"},
    {"Id_1": "2", "Me_1": "20"},
]
WRONG_ROWS: list[dict[str, str | None]] = [
    {"Id_1": "1", "Me_1": "10.5"},
    {"Id_1": "2", "Me_1": "99"},
]
EXPECTED_STRUCTURE = {
    "datasets": [
        {
            "name": "DS_r",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}
EXPECTED_CSV = "Id_1,Me_1\n1,10.5\n2,20\n"
EXPECTED_DATE = {"scalars": [{"name": "sc_r", "type": "Date", "value": None}]}


def make_operator(operator_id: str, name: str, page: str, *variants: Variant) -> Operator:
    return Operator(operator_id, name, operator_id, page, ("dataset",), ("dataset",), variants)


ALPHA = Chapter(
    "alpha",
    "Alpha operators",
    (
        make_operator(
            "first",
            "First",
            "Alpha operators/First",
            Variant("ds", "first(DS_1)", (manual("ex_1"), authored())),
            Variant("sc", "first(1)", (authored(),)),
        ),
    ),
)
BETA = Chapter(
    "beta",
    "Beta operators",
    (
        make_operator(
            "second",
            "Second",
            "Beta operators/Second",
            Variant("ds", "second(DS_1)", (authored(),)),
        ),
    ),
)
TYPES = (
    TypeDef("scalar", ()),
    TypeDef("number", ("scalar",), (Variant("identity", "identity", (authored(),)),)),
)
CAT = Catalogue((ALPHA, BETA), TYPES)
PAGES = ("Alpha operators/First", "Beta operators/Second")
MANUAL_EXAMPLES = ("Alpha operators/First/ex_1",)
KEYS = [
    "alpha/first/ds/manual_ex_1",
    "alpha/first/ds/minimal",
    "alpha/first/sc/minimal",
    "beta/second/ds/minimal",
    "types/number/identity/minimal",
]


# --- The world of a check ---


def examples_of(cat: Catalogue) -> Iterator[tuple[str, Example]]:
    """Every example of ``cat`` with its key, as the generator keys them."""
    for chapter in cat.chapters:
        for operator in chapter.operators:
            for variant in operator.variants:
                for example in variant.examples:
                    yield f"{chapter.id}/{operator.id}/{variant.id}/{example.name}", example
    for typedef in cat.types:
        for variant in typedef.variants:
            for example in variant.examples:
                yield f"types/{typedef.id}/{variant.id}/{example.name}", example


def write_examples(root: Path, cat: Catalogue) -> None:
    """The example folders of ``cat``: a script and an expected dataset (or Date scalar)."""
    for key, example in examples_of(cat):
        folder = root / "examples" / key
        (folder / "expected").mkdir(parents=True)
        (folder / "script.vtl").write_text("DS_r := DS_1;\n", encoding="utf-8")
        if example.check == "current_date":
            text = json.dumps(EXPECTED_DATE)
            (folder / "expected" / "scalars.json").write_text(text, encoding="utf-8")
        else:
            text = json.dumps(EXPECTED_STRUCTURE)
            (folder / "expected" / "DS_r.json").write_text(text, encoding="utf-8")
            (folder / "expected" / "DS_r.csv").write_text(EXPECTED_CSV, encoding="utf-8")


@dataclass
class World:
    """The generator's folder, and what the canned runner did and is to do."""

    root: Path
    modes: dict[str, str] = field(default_factory=dict)  # chapter -> good, wrong, error, diverge
    calls: list[list[tuple[Path, Engine, int]]] = field(default_factory=list)
    settings: list[tuple[int, float]] = field(default_factory=list)
    interrupt: bool = False

    def output(self, path: Path, engine: Engine) -> RunOutput:
        chapter = path.relative_to(self.root / "examples").parts[0]
        mode = self.modes.get(chapter, "good")
        if mode == "error" or (mode == "diverge" and engine == "duckdb"):
            return RunOutput(False, [], [], "SemanticError", "1-1-1-1", f"failed on {engine}")
        if (path / "expected" / "scalars.json").is_file():
            return RunOutput(True, [], [ActualScalar("sc_r", "Date", TODAY)])
        rows = WRONG_ROWS if mode == "wrong" else RIGHT_ROWS
        return RunOutput(True, [ActualDataset("DS_r", COMPONENTS, rows)], [], stable=True)

    def run_all(
        self, jobs: Sequence[tuple[Path, Engine, int]], workers: int, timeout_s: float
    ) -> dict[tuple[Path, Engine], RunOutput]:
        self.calls.append(list(jobs))
        self.settings.append((workers, timeout_s))
        if self.interrupt:
            raise KeyboardInterrupt
        return {(path, engine): self.output(path, engine) for path, engine, _ in jobs}

    def run_chapters(self) -> set[str]:
        """The chapters of the jobs that were started, in every call."""
        return {
            path.relative_to(self.root / "examples").parts[0]
            for jobs in self.calls
            for path, _, _ in jobs
        }

    def names(self) -> list[str]:
        """What is in the generator's folder."""
        return sorted(path.name for path in self.root.iterdir())

    def status(self) -> Any:
        """The document of ``status.json``."""
        return json.loads((self.root / "status.json").read_text(encoding="utf-8"))

    def index_row(self, text: str) -> list[str]:
        """The cells of the row of the chapter table of ``index.rst`` that holds ``text``."""
        page = (self.root / "index.rst").read_text(encoding="utf-8")
        chapters = re.split(r"\nChapters\n-+\n", page, maxsplit=1)[1].split("\nType lattice\n", 1)[
            0
        ]
        rows: list[list[str]] = []
        for line in chapters.splitlines():
            if line.startswith("   * -"):
                rows.append([line[7:]])
            elif line.startswith("     -") and rows:
                rows[-1].append(line[7:])
        found = [row for row in rows if any(text in cell for cell in row)]
        assert len(found) == 1, f"{len(found)} rows have {text}"
        return found[0]


def build_world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cat: Catalogue = CAT,
    pages: tuple[str, ...] = PAGES,
    manual_examples: tuple[str, ...] = MANUAL_EXAMPLES,
) -> World:
    """A generator folder with the examples of ``cat``, and the generator wired to it."""
    root = tmp_path / "operator_map"
    write_examples(root, cat)
    repo = tmp_path / "repo"
    (repo / "src" / "vtlengine").mkdir(parents=True)
    (repo / "src" / "vtlengine" / "__init__.py").write_text(
        'from vtlengine.API import run\n\n__version__ = "9.9.9"\n', encoding="utf-8"
    )
    world = World(root)
    monkeypatch.setattr(generate, "ROOT", root)
    monkeypatch.setattr(generate, "REPO", repo)
    monkeypatch.setattr(generate, "load_catalogue", lambda: cat)
    monkeypatch.setattr(generate, "run_all", world.run_all)
    monkeypatch.setattr(generate, "_today", lambda: TODAY)
    monkeypatch.setattr(generate, "_engine_commit", lambda _repo: "abc1234")
    monkeypatch.setattr(manual_index, "MANUAL_COMMIT", SPEC_COMMIT)
    monkeypatch.setattr(manual_index, "MANUAL_PAGES", pages)
    monkeypatch.setattr(manual_index, "MANUAL_EXAMPLES", manual_examples)
    return world


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return build_world(tmp_path, monkeypatch)


def problems(capsys: pytest.CaptureFixture[str]) -> list[str]:
    """The lines that the generator printed as problems."""
    return capsys.readouterr().err.splitlines()


# Every file that a run of the map ``CAT`` writes, in the order it writes them.
OUTPUTS = ["index.rst", "alpha.rst", "beta.rst", "status.json"]


def seed_outputs(world: World) -> dict[str, bytes]:
    """Old versions of every output file, with bytes that no run could give."""
    old = {name: f"old {name}, not a run \u00e9\n".encode() for name in OUTPUTS}
    for name, data in old.items():
        (world.root / name).write_bytes(data)
    return old


def assert_untouched(world: World, old: dict[str, bytes]) -> None:
    """Every old file is as it was, and nothing else (a staging folder) is left."""
    assert {name: (world.root / name).read_bytes() for name in old} == old
    assert world.names() == sorted([*old, "examples"])


# --- Validation ---


def test_failed_validation_writes_nothing(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    old = seed_outputs(world)
    shutil.rmtree(world.root / "examples/beta/second/ds/minimal")
    assert generate.main([]) == 1
    # Every output file is as it was, nothing else was written, not even a temporary folder,
    # and nothing was run.
    assert_untouched(world, old)
    assert world.calls == []
    assert "missing example folder: examples/beta/second/ds/minimal" in problems(capsys)


def test_problems_of_the_catalogue_are_printed_one_per_line(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = make_operator("odd", "Odd", "Alpha operators/Odd", Variant("ds", "odd", (authored(),)))
    cat = Catalogue((Chapter("alpha", "Alpha operators", (*ALPHA.operators, bad)), BETA), TYPES)
    monkeypatch.setattr(generate, "load_catalogue", lambda: cat)
    assert generate.main([]) == 1
    lines = problems(capsys)
    assert "alpha.odd: unknown page 'Alpha operators/Odd'" in lines
    # The folder of the new example is missing too: every problem is on a line of its own.
    assert "missing example folder: examples/alpha/odd/ds/minimal" in lines
    assert all(line for line in lines)
    assert world.names() == ["examples"]


def test_example_folder_that_is_malformed_is_a_problem(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    (world.root / "examples/beta/second/ds/minimal/expected/DS_r.csv").unlink()
    assert generate.main([]) == 1
    assert any("DS_r.csv" in line for line in problems(capsys))
    assert world.calls == []


def test_example_folder_without_a_script_is_not_called_missing(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    (world.root / "examples/beta/second/ds/minimal/script.vtl").unlink()
    assert generate.main([]) == 1
    [line] = problems(capsys)
    assert "missing script.vtl" in line
    assert "missing example folder" not in line
    assert world.calls == []


def test_orphan_example_folder_fails(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    for stray in ("alpha/first/ds/stray", "types/number/identity/extra"):
        folder = world.root / "examples" / stray
        folder.mkdir(parents=True)
        (folder / "script.vtl").write_text("DS_r := DS_1;\n", encoding="utf-8")
    # A folder that is not an example (deeper, or with no script) is not an orphan.
    (world.root / "examples/alpha/first/ds/minimal/extra").mkdir()
    (world.root / "examples/alpha/first/ds/empty").mkdir()
    orphans = [
        "example folder not in the catalogue: examples/alpha/first/ds/stray",
        "example folder not in the catalogue: examples/types/number/identity/extra",
    ]
    assert generate.main([]) == 1
    assert problems(capsys) == orphans
    # The orphans are checked whatever --only says, in a chapter that was not named too.
    assert generate.main(["--only", "beta", "--check-only"]) == 1
    assert problems(capsys) == orphans
    assert world.calls == []
    assert world.names() == ["examples"]


def test_missing_folders_are_checked_only_for_the_chapters_named(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.rmtree(world.root / "examples/beta/second/ds/minimal")
    assert generate.main(["--only", "alpha", "--check-only"]) == 0
    assert generate.main(["--only", "types", "--check-only"]) == 0
    capsys.readouterr()
    assert generate.main(["--only", "beta", "--check-only"]) == 1
    assert problems(capsys) == ["missing example folder: examples/beta/second/ds/minimal"]
    # The types count as a chapter.
    shutil.rmtree(world.root / "examples/types/number/identity/minimal")
    assert generate.main(["--only", "types", "--check-only"]) == 1
    assert problems(capsys) == ["missing example folder: examples/types/number/identity/minimal"]


def test_only_limits_the_page_coverage_to_the_chapters_named(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pages = (*PAGES, "Alpha operators/Extra", "Beta operators/Extra")
    monkeypatch.setattr(manual_index, "MANUAL_PAGES", pages)
    assert generate.main(["--check-only"]) == 1
    assert problems(capsys) == [
        "page not covered: Alpha operators/Extra",
        "page not covered: Beta operators/Extra",
    ]
    assert generate.main(["--only", "alpha", "--check-only"]) == 1
    assert problems(capsys) == ["page not covered: Alpha operators/Extra"]
    assert generate.main(["--only", "types", "--check-only"]) == 0


def test_unknown_only_chapter_is_rejected(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    assert generate.main(["--only", "alpha", "--only", "nope"]) == 1
    lines = problems(capsys)
    assert len(lines) == 1
    assert "unknown chapter 'nope'" in lines[0]
    assert "alpha, beta, types" in lines[0]
    assert world.calls == []
    assert world.names() == ["examples"]


def test_check_only_checks_and_writes_nothing(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate.main(["--check-only"]) == 0
    assert capsys.readouterr().err == ""
    assert world.calls == []
    assert world.names() == ["examples"]


def test_a_commit_that_cannot_be_read_is_a_problem(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_git(_repo: Path) -> str:
        raise generate.GenerateError("cannot run git in the worktree")

    monkeypatch.setattr(generate, "_engine_commit", no_git)
    assert generate.main([]) == 1
    assert problems(capsys) == ["cannot run git in the worktree"]
    assert world.calls == []
    assert world.names() == ["examples"]


@pytest.mark.parametrize("option", ["--workers", "--timeout"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "many"])
def test_workers_and_timeout_must_be_finite_numbers_above_zero(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as stopped:
        generate.main([option, value])
    assert stopped.value.code == 2


# --- The jobs ---


def test_jobs_are_every_example_on_both_engines_and_random_runs_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = make_operator(
        "rnd",
        "Random",
        "Alpha operators/Random",
        Variant("ds", "random(DS_1, 1)", (authored(check="random"),)),
    )
    alpha = Chapter("alpha", "Alpha operators", (*ALPHA.operators, extra))
    cat = Catalogue((alpha, BETA), TYPES)
    world = build_world(tmp_path, monkeypatch, cat, (*PAGES, "Alpha operators/Random"))
    assert generate.main([]) == 0
    base = world.root / "examples"
    [jobs] = world.calls
    assert jobs == [
        (base / folder, engine, 2 if folder == "alpha/rnd/ds/minimal" else 1)
        for folder in (*KEYS[:3], "alpha/rnd/ds/minimal", *KEYS[3:])
        for engine in ("pandas", "duckdb")
    ]
    assert world.settings == [(4, 120.0)]


def test_workers_and_timeout_reach_the_runner(world: World) -> None:
    assert generate.main(["--workers", "2", "--timeout", "30"]) == 0
    assert world.settings == [(2, 30.0)]


def test_only_runs_the_jobs_of_the_chapters_named(world: World) -> None:
    assert generate.main([]) == 0
    world.calls.clear()
    assert generate.main(["--only", "types", "--only", "beta"]) == 0
    assert world.run_chapters() == {"types", "beta"}
    [jobs] = world.calls
    assert len(jobs) == 4  # two examples on two engines


def test_current_date_is_checked_against_the_day_of_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = make_operator(
        "now",
        "Now",
        "Alpha operators/Now",
        Variant("sc", "current_date()", (authored(check="current_date"),)),
    )
    cat = Catalogue((Chapter("alpha", "Alpha operators", (today,)),), ())
    world = build_world(tmp_path, monkeypatch, cat, ("Alpha operators/Now",), ())
    assert generate.main([]) == 0
    assert world.status()["examples"]["alpha/now/sc/minimal"]["status"] == "ok"
    # The same output on another day is not today.
    monkeypatch.setattr(generate, "_today", lambda: "2026-10-24")
    assert generate.main([]) == 0
    assert world.status()["examples"]["alpha/now/sc/minimal"]["status"] == "partial"


def test_keyboard_interrupt_exits_2_and_writes_nothing(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    old = seed_outputs(world)
    world.interrupt = True
    assert generate.main([]) == 2
    assert "interrupted" in capsys.readouterr().err
    assert_untouched(world, old)


# --- Merging with the previous run ---


def test_only_merges_previous_status(world: World) -> None:
    world.modes = {"beta": "wrong"}
    assert generate.main([]) == 0
    beta = [":doc:`Beta operators <beta>`", "🟡", "1", "0", "1", "0", "0"]
    assert world.index_row("<beta>") == beta
    assert world.index_row("<alpha>") == [
        ":doc:`Alpha operators <alpha>`",
        "🟢",
        "2",
        "2",
        "0",
        "0",
        "0",
    ]
    # A rerun of beta would now give 🟢, and alpha is 🔴: only alpha is run.
    world.modes = {"alpha": "error"}
    world.calls.clear()
    assert generate.main(["--only", "alpha"]) == 0
    assert world.run_chapters() == {"alpha"}
    assert world.index_row("<alpha>") == [
        ":doc:`Alpha operators <alpha>`",
        "🔴",
        "2",
        "0",
        "0",
        "2",
        "0",
    ]
    assert world.index_row("<beta>") == beta
    assert world.index_row("Data types") == ["Data types", "🟢", "1", "1", "0", "0", "0"]
    examples = world.status()["examples"]
    assert examples["beta/second/ds/minimal"]["status"] == "partial"
    assert examples["alpha/first/sc/minimal"]["status"] == "missing"
    assert sorted(examples) == sorted(KEYS)
    assert (world.root / "beta.rst").is_file()


def test_only_drops_the_results_of_examples_that_left_the_catalogue(world: World) -> None:
    assert generate.main([]) == 0
    document = world.status()
    document["examples"]["beta/second/ds/gone"] = document["examples"]["beta/second/ds/minimal"]
    (world.root / "status.json").write_text(json.dumps(document), encoding="utf-8")
    assert generate.main(["--only", "alpha"]) == 0
    assert sorted(world.status()["examples"]) == sorted(KEYS)


def test_only_without_a_previous_result_is_a_validation_problem(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate.main(["--only", "alpha"]) == 1
    lines = problems(capsys)
    assert sorted(line.split(":")[0] for line in lines) == [
        "no result for beta/second/ds/minimal",
        "no result for types/number/identity/minimal",
    ]
    assert world.calls == []
    assert world.names() == ["examples"]


def test_only_with_an_unreadable_status_file_is_one_problem(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    (world.root / "status.json").write_text("{not json", encoding="utf-8")
    assert generate.main(["--only", "alpha"]) == 1
    lines = problems(capsys)
    assert len(lines) == 1
    assert lines[0].startswith("status.json: cannot read the previous results")
    assert world.calls == []


def test_only_naming_every_example_chapter_does_not_read_the_previous_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No type has an example: the chapters are all there is, and ``--only`` names both.
    cat = Catalogue((ALPHA, BETA), (TypeDef("scalar", ()),))
    world = build_world(tmp_path, monkeypatch, cat)
    (world.root / "status.json").write_text("{not json", encoding="utf-8")
    assert generate.main(["--only", "alpha", "--only", "beta"]) == 0
    assert sorted(world.status()["examples"]) == sorted(KEYS[:4])


def test_only_naming_every_chapter_and_the_types_does_not_read_the_previous_results(
    world: World,
) -> None:
    (world.root / "status.json").write_text("{not json", encoding="utf-8")
    assert generate.main(["--only", "alpha", "--only", "beta", "--only", "types"]) == 0
    assert sorted(world.status()["examples"]) == sorted(KEYS)


def test_only_reads_the_previous_results_when_a_chapter_with_examples_is_left_out(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    # The types have examples, so naming the two chapters is not naming everything.
    (world.root / "status.json").write_text("{not json", encoding="utf-8")
    assert generate.main(["--only", "alpha", "--only", "beta"]) == 1
    [line] = problems(capsys)
    assert line.startswith("status.json: cannot read the previous results")


@pytest.mark.parametrize(("variants", "more"), [(8, None), (9, 1), (13, 5)])
def test_only_prints_ten_no_result_lines_at_most_and_counts_the_rest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    variants: int,
    more: int | None,
) -> None:
    forms = tuple(Variant(f"v{n}", f"many(DS_{n})", (authored(),)) for n in range(variants))
    many = make_operator("many", "Many", "Gamma operators/Many", *forms)
    cat = Catalogue((ALPHA, BETA, Chapter("gamma", "Gamma operators", (many,))), TYPES)
    world = build_world(tmp_path, monkeypatch, cat, (*PAGES, "Gamma operators/Many"))
    assert generate.main(["--only", "alpha"]) == 1
    lines = problems(capsys)
    # Beta, the variants of gamma and the type example have no result: 2 + ``variants`` examples.
    expected = ["beta/second/ds/minimal", *(f"gamma/many/v{n}/minimal" for n in range(variants))]
    expected.append("types/number/identity/minimal")
    assert [line.split(":")[0] for line in lines[:10]] == [
        f"no result for {x}" for x in expected[:10]
    ]
    if more is None:
        assert len(lines) == 10
    else:
        assert lines[10:] == [f"… and {more} more"]
    assert world.calls == []


# --- What is written ---


def test_the_run_writes_every_file_and_leaves_nothing_else(world: World) -> None:
    assert generate.main([]) == 0
    # Sphinx pages and the results, and no Markdown file.
    assert world.names() == ["alpha.rst", "beta.rst", "examples", "index.rst", "status.json"]
    index = (world.root / "index.rst").read_text(encoding="utf-8")
    rule = "=" * 40
    assert index.startswith(f"{rule}\nVTL 2.2 operator map\n{rule}\n")
    assert ".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n   alpha\n   beta\n" in index
    assert "9.9.9" in index
    assert "abc1234" in index
    assert f"(sdmx-twg/vtl @ {SPEC_COMMIT})" in index
    alpha = (world.root / "alpha.rst").read_text(encoding="utf-8")
    rule = "=" * 34  # twice the 17 characters of "🟢 Alpha operators"
    assert alpha.startswith(f"{rule}\n🟢 Alpha operators\n{rule}\n")
    assert alpha.count("/script.vtl>`__") == 3
    assert ".. mermaid::\n\n   flowchart LR\n" in alpha


def test_status_json_has_sorted_keys_the_meta_and_every_verdict(world: World) -> None:
    world.modes = {"alpha": "diverge"}
    assert generate.main([]) == 0
    text = (world.root / "status.json").read_text(encoding="utf-8")
    document = json.loads(text)
    assert text == json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert document["meta"] == {
        "date": TODAY,
        "spec_commit": SPEC_COMMIT,
        "vtlengine_commit": "abc1234",
        "vtlengine_version": "9.9.9",
    }
    assert sorted(document["examples"]) == sorted(KEYS)
    assert document["examples"]["alpha/first/sc/minimal"] == {
        "duckdb": {"detail": "SemanticError 1-1-1-1: failed on duckdb", "outcome": "error"},
        "engines_agree": False,
        "pandas": {"detail": "", "outcome": "match"},
        "status": "diverge",
    }
    assert document["examples"]["beta/second/ds/minimal"]["status"] == "ok"


def test_a_rerun_gives_the_same_files(world: World) -> None:
    assert generate.main([]) == 0
    first = {name: (world.root / name).read_bytes() for name in world.names() if name != "examples"}
    assert generate.main([]) == 0
    assert {name: (world.root / name).read_bytes() for name in first} == first


def test_a_block_over_a_limit_exits_1_and_writes_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def too_big(*_args: object, **_kwargs: object) -> str:
        raise MermaidLimitError("type flow of dataset: 501 edges, and a block needs fewer than 500")

    monkeypatch.setattr(generate, "index_rst", too_big)
    old = seed_outputs(world)
    assert generate.main([]) == 1
    assert problems(capsys) == ["type flow of dataset: 501 edges, and a block needs fewer than 500"]
    assert_untouched(world, old)


def test_every_block_of_the_rendered_files_is_checked_against_the_limits(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    edges = "\n".join(f"       a{n} --> b{n}" for n in range(500))
    big = f"Chapter\n=======\n\n.. mermaid::\n\n   flowchart LR\n{edges}\n\ntext\n"
    monkeypatch.setattr(generate, "chapter_rst", lambda *_args, **_kwargs: big)
    old = seed_outputs(world)
    assert generate.main([]) == 1
    [line] = problems(capsys)
    assert line.startswith("alpha.rst: ")
    assert "500 edges" in line
    assert_untouched(world, old)


# --- Internal errors and failed writes ---


def test_an_internal_error_exits_2_with_one_line(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("something is\nwrong inside")

    old = seed_outputs(world)
    monkeypatch.setattr(generate, "build_index", broken)
    assert generate.main([]) == 2
    # One line: the type and the message, with the message on one line too.
    assert problems(capsys) == ["internal error: RuntimeError: something is wrong inside"]
    assert_untouched(world, old)


def break_the_staging_write(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A text that cannot be encoded: the write of the second file to be written raises."""
    monkeypatch.setattr(generate, "chapter_rst", lambda *_a, **_k: "lone \udc80 surrogate\n")


def break_the_replace(nth: int) -> Callable[[World, pytest.MonkeyPatch], None]:
    """The ``nth`` ``os.replace`` of the run raises; the others do what they are asked."""

    def inject(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
        real_replace = os.replace
        calls: list[object] = []

        def flaky(source: Any, target: Any) -> None:
            calls.append(target)
            if len(calls) == nth:
                raise OSError(errno.EACCES, "cannot replace")
            real_replace(source, target)

        monkeypatch.setattr(os, "replace", flaky)

    return inject


@pytest.mark.parametrize("seeded", [True, False], ids=["old files", "no old files"])
@pytest.mark.parametrize(
    "inject",
    [break_the_staging_write, break_the_replace(1), break_the_replace(3)],
    ids=["staging write", "first replace", "third replace"],
)
def test_a_write_that_fails_leaves_the_old_files_and_no_staging_folder(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    inject: Callable[[World, pytest.MonkeyPatch], None],
    seeded: bool,
) -> None:
    old = seed_outputs(world) if seeded else {}
    inject(world, monkeypatch)
    assert generate.main([]) == 2
    [line] = problems(capsys)
    assert line.startswith("internal error: ")
    # The files that were moved before the failure are put back, or removed when they are new.
    assert_untouched(world, old)


def interrupt() -> None:
    raise KeyboardInterrupt


def run_main_interrupted() -> int:
    """``generate.main`` of a run that is interrupted: an interrupt that gets out is a failure."""
    try:
        return generate.main([])
    except KeyboardInterrupt:
        pytest.fail("the interrupt got out of main")


def interrupt_judging(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate, "_judge", lambda *_a, **_k: interrupt())


def interrupt_rendering(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate, "chapter_rst", lambda *_a, **_k: interrupt())


def interrupt_the_replace(
    nth: int, *, after: bool = False
) -> Callable[[World, pytest.MonkeyPatch], None]:
    """The ``nth`` ``os.replace`` is interrupted: before it moves the file, or right after."""

    def inject(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
        real_replace = os.replace
        calls: list[object] = []

        def flaky(source: Any, target: Any) -> None:
            calls.append(target)
            if len(calls) == nth and not after:
                interrupt()
            real_replace(source, target)
            if len(calls) == nth:
                interrupt()

        monkeypatch.setattr(os, "replace", flaky)

    return inject


@pytest.mark.parametrize("seeded", [True, False], ids=["old files", "no old files"])
@pytest.mark.parametrize(
    "inject",
    [
        interrupt_judging,
        interrupt_rendering,
        interrupt_the_replace(1),
        interrupt_the_replace(3),
        interrupt_the_replace(1, after=True),
        interrupt_the_replace(3, after=True),
    ],
    ids=[
        "judging",
        "rendering",
        "first replace",
        "third replace",
        "right after the first replace",
        "right after the third replace",
    ],
)
def test_an_interrupt_at_any_stage_exits_2_and_leaves_the_old_files(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    inject: Callable[[World, pytest.MonkeyPatch], None],
    seeded: bool,
) -> None:
    old = seed_outputs(world) if seeded else {}
    inject(world, monkeypatch)
    assert run_main_interrupted() == 2
    assert "interrupted" in capsys.readouterr().err
    # A move that was cut short, right after it was done, is undone as well.
    assert_untouched(world, old)


def staging_folders(world: World) -> list[Path]:
    """The staging folders of the generator that are left in its folder."""
    return [path for path in world.root.iterdir() if path.name.startswith(".generate-")]


def failure(kind: type[BaseException]) -> BaseException:
    """What a stubbed ``os.replace`` raises: a failure of the system, or an interrupt."""
    return OSError(errno.EACCES, "no way") if kind is OSError else KeyboardInterrupt()


@pytest.mark.parametrize(
    "restore_fails", [OSError, KeyboardInterrupt], ids=["failed restore", "second interrupt"]
)
@pytest.mark.parametrize(
    "replace_fails", [OSError, KeyboardInterrupt], ids=["failed replace", "interrupted replace"]
)
def test_a_restore_that_fails_keeps_the_staging_folder_and_the_others_are_still_tried(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    replace_fails: type[BaseException],
    restore_fails: type[BaseException],
) -> None:
    old = seed_outputs(world)
    real_replace = os.replace
    calls: list[tuple[str, str]] = []  # (the name of the file, the folder it comes from)

    def flaky(source: Any, target: Any) -> None:
        file, folder = Path(source).name, Path(source).parent.name
        calls.append((file, folder))
        restores = [name for name, where in calls if where == "old"]
        # The third file cannot be replaced, and the first file to be put back cannot be put back,
        # for a failure of the system or because the run is interrupted again.
        if len(calls) == 3:
            raise failure(replace_fails)
        if folder == "old" and len(restores) == 1:
            raise failure(restore_fails)
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", flaky)
    assert generate.main([]) == 2
    [line] = problems(capsys)
    failed = next(name for name, where in calls if where == "old")
    # Every file that was replaced, the one that could not be replaced too, was tried.
    assert sorted(name for name, where in calls if where == "old") == sorted(OUTPUTS[:3])
    # The file that could not be put back has the new bytes; the staging folder has the old ones.
    [staging] = staging_folders(world)
    assert (staging / "old" / failed).read_bytes() == old[failed]
    assert (world.root / failed).read_bytes() != old[failed]
    # The other files are as they were, and the line says where the old files are.
    others = {name: data for name, data in old.items() if name != failed}
    assert {name: (world.root / name).read_bytes() for name in others} == others
    assert line.startswith("internal error: ")
    assert str(staging) in line
    assert failed in line


def new_files_are_in_place(world: World, old: dict[str, bytes]) -> bool:
    """Whether every output file is a new one: none has the old bytes."""
    return all((world.root / name).read_bytes() != data for name, data in old.items())


def test_an_interrupt_while_the_staging_folder_is_removed_says_that_the_files_are_written(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    old = seed_outputs(world)
    monkeypatch.setattr(shutil, "rmtree", lambda *_a, **_k: interrupt())
    assert run_main_interrupted() == 2
    [line] = problems(capsys)
    # Every new file was in place before the clean-up: nothing is as it was, and the line says
    # so, and where the folder that the clean-up did not remove is.
    assert new_files_are_in_place(world, old)
    [staging] = staging_folders(world)
    assert line == (
        "interrupted after the files were written: every new file is in place; "
        f"the folder {staging} was left over, remove it"
    )


def test_an_interrupt_in_the_summary_says_that_the_files_are_written(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    old = seed_outputs(world)
    monkeypatch.setattr(generate, "counts", lambda *_a, **_k: interrupt())
    assert run_main_interrupted() == 2
    assert problems(capsys) == [
        "interrupted after the files were written: every new file is in place"
    ]
    assert new_files_are_in_place(world, old)
    assert staging_folders(world) == []


def test_an_error_in_the_summary_says_that_the_files_are_written(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("boom")

    old = seed_outputs(world)
    monkeypatch.setattr(generate, "counts", broken)
    assert generate.main([]) == 2
    assert problems(capsys) == ["internal error: RuntimeError: boom; every new file is in place"]
    assert new_files_are_in_place(world, old)


def test_an_interrupt_while_the_folder_is_removed_after_a_rollback_names_the_folder_too(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The first interrupt is at the third move and the rollback works; the second one comes while
    # the staging folder is removed. The old files are as they were, and a folder is left over.
    old = seed_outputs(world)
    interrupt_the_replace(3)(world, monkeypatch)
    monkeypatch.setattr(shutil, "rmtree", lambda *_a, **_k: interrupt())
    assert run_main_interrupted() == 2
    [line] = problems(capsys)
    [staging] = staging_folders(world)
    assert {name: (world.root / name).read_bytes() for name in old} == old
    assert line == (
        f"interrupted: nothing was written; the folder {staging} was left over, remove it"
    )


def test_a_new_file_that_cannot_be_removed_keeps_the_staging_folder_too(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No old files: undoing the first replaces is removing the new files, and that fails.
    real_replace = os.replace
    calls: list[object] = []

    def third_fails(source: Any, target: Any) -> None:
        calls.append(target)
        if len(calls) == 3:
            raise OSError(errno.EACCES, "no way")
        real_replace(source, target)

    def cannot_remove(self: Path, missing_ok: bool = False) -> None:
        raise OSError(errno.EACCES, "no way")

    monkeypatch.setattr(os, "replace", third_fails)
    monkeypatch.setattr(Path, "unlink", cannot_remove)
    assert generate.main([]) == 2
    [line] = problems(capsys)
    [staging] = staging_folders(world)
    assert str(staging) in line
    assert "index.rst" in line


# --- The metadata of the run ---


def test_engine_version_is_read_without_importing_the_engine(tmp_path: Path) -> None:
    init = tmp_path / "src" / "vtlengine" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text(
        'from vtlengine.API import run\n\n__all__ = ["run"]\n__version__ = "1.2.3rc4"\n',
        encoding="utf-8",
    )
    assert generate._engine_version(tmp_path) == "1.2.3rc4"
    init.write_text("__version__: str = version()\n", encoding="utf-8")
    with pytest.raises(generate.GenerateError, match="__version__"):
        generate._engine_version(tmp_path)


def test_engine_commit_is_the_short_hash_of_the_worktree() -> None:
    assert re.fullmatch(r"[0-9a-f]{7,40}", generate._engine_commit(generate.REPO))


def test_the_generator_does_not_import_the_engine_or_pandas() -> None:
    code = (
        "import sys, generate; sys.exit(int('vtlengine' in sys.modules or 'pandas' in sys.modules))"
    )
    operator_map = str(Path(generate.__file__).resolve().parent)
    env = {**os.environ, "PYTHONPATH": operator_map}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)  # noqa: S603


def test_the_default_folders_are_the_ones_of_this_worktree() -> None:
    here = Path(generate.__file__).resolve().parent
    assert here == generate.ROOT
    assert generate.ROOT == generate.REPO / "docs" / "operator_map"
    assert (generate.REPO / "src" / "vtlengine" / "__init__.py").is_file()
