# ruff: noqa: S101
"""Checks for the generator: validation, the jobs it starts, merging, and the files it writes.

The runner is replaced by canned outputs, the catalogue and the manual index by small hand-made
ones, and the folder that the generator writes to by a temporary one: no engine runs here. Each
check gets a map of two chapters, with their example folders on disk.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import generate
import pytest
from catalogue import manual_index

from tool import runner
from tool.compare import ActualComponent, ActualDataset, ActualScalar
from tool.mermaid import MermaidLimitError
from tool.model import (
    Catalogue,
    Chapter,
    Example,
    Operator,
    Variant,
    authored,
    load_catalogue,
    manual,
)
from tool.runner import Engine, RunOutput
from tool.status import EngineVerdict

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
# The notes of an example in the fixed layout.
NOTES = ":Tests: What it tests.\n:Manual: The rule.\n:Expected: The result.\n"
# The engines that the preflight child of every run reports.
PANDAS_VERSION = "2.3.3"
DUCKDB_VERSION = "1.5.5"


def make_operator(operator_id: str, name: str, page: str, *variants: Variant) -> Operator:
    return Operator(operator_id, name, operator_id, page, ("dataset",), ("dataset",), variants)


def form(variant_id: str, label: str, *examples: Example) -> Variant:
    """A form that declares the types it takes and gives, as the checks of the catalogue ask."""
    return Variant(variant_id, label, examples, takes=("number",), gives=("number",))


ALPHA = Chapter(
    "alpha",
    "Alpha operators",
    (
        make_operator(
            "first",
            "First",
            "Alpha operators/First",
            form("ds", "first(DS_1)", manual("ex_1"), authored("two_rows")),
            form("sc", "first(1)", authored("one_value")),
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
            form("ds", "second(DS_1)", authored("two_rows")),
        ),
    ),
)
CAT = Catalogue((ALPHA, BETA))
PAGES = ("Alpha operators/First", "Beta operators/Second")
MANUAL_EXAMPLES = ("Alpha operators/First/ex_1",)
KEYS = [
    "alpha/first/ds/manual_ex_1",
    "alpha/first/ds/two_rows",
    "alpha/first/sc/one_value",
    "beta/second/ds/two_rows",
]


# --- The world of a check ---


def examples_of(cat: Catalogue) -> Iterator[tuple[str, Example]]:
    """Every example of ``cat`` with its key, as the generator keys them."""
    for chapter in cat.chapters:
        for operator in chapter.operators:
            for variant in operator.variants:
                for example in variant.examples:
                    yield f"{chapter.id}/{operator.id}/{variant.id}/{example.name}", example


def write_examples(root: Path, cat: Catalogue) -> None:
    """The example folders of ``cat``: a script, its notes, and an expected dataset (or scalar)."""
    for key, example in examples_of(cat):
        folder = root / "examples" / key
        (folder / "expected").mkdir(parents=True)
        (folder / "script.vtl").write_text("DS_r := DS_1;\n", encoding="utf-8")
        (folder / "notes.rst").write_text(NOTES, encoding="utf-8")
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
    # chapter -> good, wrong, error (not implemented), refused (on purpose), diverge
    modes: dict[str, str] = field(default_factory=dict)
    calls: list[list[tuple[Path, Engine, int]]] = field(default_factory=list)
    settings: list[tuple[int, float]] = field(default_factory=list)
    interrupt: bool = False
    preflights: list[float] = field(default_factory=list)  # the timeout of each preflight

    def preflight(self, timeout_s: float) -> Any:
        self.preflights.append(timeout_s)
        return runner.EngineVersions(PANDAS_VERSION, DUCKDB_VERSION)

    def output(self, path: Path, engine: Engine) -> RunOutput:
        chapter = path.relative_to(self.root / "examples").parts[0]
        mode = self.modes.get(chapter, "good")
        if mode == "error":  # a gap: the error says that the construct is not supported
            return RunOutput(False, [], [], "SemanticError", "1-3-13", f"failed on {engine}")
        if mode == "refused":  # on purpose: an error of the engine about the values
            return RunOutput(False, [], [], "RunTimeError", "2-1-15-6", f"refused on {engine}")
        if mode == "diverge" and engine == "duckdb":
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
        """What is in the generator's folder, at its top level."""
        return sorted(path.name for path in self.root.iterdir())

    def tree(self) -> list[str]:
        """Everything under the generator's folder but the examples, as paths: folders end in /.

        A staging folder that was left over is in it too, with what it holds.
        """
        found = []
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root).as_posix()
            if relative != "examples" and not relative.startswith("examples/"):
                found.append(relative + "/" if path.is_dir() else relative)
        return sorted(found)

    def files(self) -> dict[str, bytes]:
        """The bytes of every file that ``tree`` lists."""
        return {
            name: (self.root / name).read_bytes() for name in self.tree() if not name.endswith("/")
        }

    def status(self) -> Any:
        """The document of ``status.json``."""
        return json.loads((self.root / "status.json").read_text(encoding="utf-8"))

    def index_row(self, text: str) -> list[str]:
        """The cells of the row of the block table of ``overview.rst`` that holds ``text``."""
        page = (self.root / "overview.rst").read_text(encoding="utf-8")
        chapters = page.split("\n.. rubric:: Operators\n", 1)[1]
        chapters = chapters.split("\n.. rubric:: ", 1)[0]
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
    monkeypatch.setattr(generate, "preflight", world.preflight, raising=False)
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


# Every file that a run of the map ``CAT`` writes, in the order it writes them, and then the
# manifest that lists them, which it writes last.
OUTPUTS = [
    "index.rst",
    "how_to_read.rst",
    "overview.rst",
    "operators.rst",
    "decisions.rst",
    "alpha/index.rst",
    "alpha/first.rst",
    "beta/index.rst",
    "beta/second.rst",
    "run.html",
    "status.json",
    "example_pages.json",
]
MANIFEST = ".generated"
# The files of an old run that no run of ``CAT`` generates: one in the folder of an operator that
# is still there, and two in a folder that is gone.
STALE = ["alpha/removed.rst", "gone/index.rst", "gone/page.rst"]


def folders_of(names: Iterable[str]) -> list[str]:
    """The folders that hold ``names``, each as ``tree`` lists it."""
    return sorted(
        {
            "/".join(parts[:n]) + "/"
            for parts in (name.split("/") for name in names)
            for n in range(1, len(parts))
        }
    )


def seed_outputs(world: World) -> dict[str, bytes]:
    """Old versions of every output file, and old files that no run writes any more.

    The bytes are such that no run could give them, and the old manifest lists every file.
    """
    old = {name: f"old {name}, not a run \u00e9\n".encode() for name in [*OUTPUTS, *STALE]}
    old[MANIFEST] = "".join(f"{name}\n" for name in sorted([*OUTPUTS, *STALE])).encode()
    for name, data in old.items():
        (world.root / name).parent.mkdir(parents=True, exist_ok=True)
        (world.root / name).write_bytes(data)
    return old


def assert_untouched(world: World, old: dict[str, bytes]) -> None:
    """Every old file is as it was, and nothing else is left: no file, no folder, no staging."""
    assert {name: (world.root / name).read_bytes() for name in old} == old
    assert world.tree() == sorted([*old, *folders_of(old)])


def staged(source: Any) -> tuple[str, str] | None:
    """Where a file that is moved is in the staging folder: ``("new" or "old", its name)``.

    A name has the folders of a block in it (``alpha/index.rst``); a file that is not in a
    staging folder gives None.
    """
    parts = Path(source).parts
    for place, part in enumerate(parts):
        if part.startswith(".generate-"):
            return parts[place + 1], "/".join(parts[place + 2 :])
    return None


# --- Validation ---


def test_failed_validation_writes_nothing(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    old = seed_outputs(world)
    shutil.rmtree(world.root / "examples/beta/second/ds/two_rows")
    assert generate.main([]) == 1
    # Every output file is as it was, nothing else was written, not even a temporary folder,
    # and nothing was run.
    assert_untouched(world, old)
    assert world.calls == []
    assert world.preflights == []  # no child starts for a run that cannot start
    assert "missing example folder: examples/beta/second/ds/two_rows" in problems(capsys)


def test_problems_of_the_catalogue_are_printed_one_per_line(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = make_operator("odd", "Odd", "Alpha operators/Odd", form("ds", "odd", authored("one_row")))
    cat = Catalogue((Chapter("alpha", "Alpha operators", (*ALPHA.operators, bad)), BETA))
    monkeypatch.setattr(generate, "load_catalogue", lambda: cat)
    assert generate.main([]) == 1
    lines = problems(capsys)
    assert "alpha.odd: unknown page 'Alpha operators/Odd'" in lines
    # The folder of the new example is missing too: every problem is on a line of its own.
    assert "missing example folder: examples/alpha/odd/ds/one_row" in lines
    assert all(line for line in lines)
    assert world.names() == ["examples"]


def test_example_folder_that_is_malformed_is_a_problem(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    (world.root / "examples/beta/second/ds/two_rows/expected/DS_r.csv").unlink()
    assert generate.main([]) == 1
    assert any("DS_r.csv" in line for line in problems(capsys))
    assert world.calls == []


def test_example_folder_without_a_script_is_not_called_missing(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    (world.root / "examples/beta/second/ds/two_rows/script.vtl").unlink()
    assert generate.main([]) == 1
    [line] = problems(capsys)
    assert "missing script.vtl" in line
    assert "missing example folder" not in line
    assert world.calls == []


def test_orphan_example_folder_fails(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    # The second one is in a folder that is no chapter, as the type hubs of the first version of
    # the map were.
    for stray in ("alpha/first/ds/stray", "types/number/identity/minimal"):
        folder = world.root / "examples" / stray
        folder.mkdir(parents=True)
        (folder / "script.vtl").write_text("DS_r := DS_1;\n", encoding="utf-8")
    # A folder that is not an example (deeper, or with no script) is not an orphan.
    (world.root / "examples/alpha/first/ds/two_rows/extra").mkdir()
    (world.root / "examples/alpha/first/ds/empty").mkdir()
    orphans = [
        "example folder not in the catalogue: examples/alpha/first/ds/stray",
        "example folder not in the catalogue: examples/types/number/identity/minimal",
    ]
    assert generate.main([]) == 1
    assert problems(capsys) == orphans
    # The orphans are checked whatever --only says, in a chapter that was not named too.
    assert generate.main(["--only", "beta", "--check-only"]) == 1
    assert problems(capsys) == orphans
    assert world.calls == []
    assert world.names() == ["examples"]


def test_missing_folders_are_checked_for_every_chapter(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    # Every run renders the pages of every chapter from their folders: ``--only`` leaves none of
    # them out of the checks.
    shutil.rmtree(world.root / "examples/beta/second/ds/two_rows")
    missing = ["missing example folder: examples/beta/second/ds/two_rows"]
    for only in ([], ["--only", "alpha"], ["--only", "beta"]):
        assert generate.main(["--check-only", *only]) == 1
        assert problems(capsys) == missing


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
    assert generate.main(["--only", "beta", "--check-only"]) == 1
    assert problems(capsys) == ["page not covered: Beta operators/Extra"]


def test_unknown_only_chapter_is_rejected(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    assert generate.main(["--only", "alpha", "--only", "nope"]) == 1
    lines = problems(capsys)
    assert len(lines) == 1
    assert "unknown chapter 'nope'" in lines[0]
    assert "(known: alpha, beta)" in lines[0]
    # The type hubs of the first version of the map are gone, and their group with them.
    assert generate.main(["--only", "types", "--check-only"]) == 1
    assert problems(capsys) == ["unknown chapter 'types' for --only (known: alpha, beta)"]
    assert world.calls == []
    assert world.names() == ["examples"]


def test_check_only_checks_and_writes_nothing(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate.main(["--check-only"]) == 0
    assert capsys.readouterr().err == ""
    assert world.calls == []
    assert world.preflights == []
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


# --- The checks of the forms, the names and the notes: every run, every chapter ---


LAYOUT = (
    "they must be Tests, Manual, Expected, then any of Finding, Erratum, Adaptation, "
    "Team decision, in this order and once each"
)


def untyped_catalogue() -> Catalogue:
    """The map of ``CAT`` as it was first written: untyped forms, examples named ``minimal``."""
    first = make_operator(
        "first",
        "First",
        "Alpha operators/First",
        Variant("ds", "first(DS_1)", (manual("ex_1"), authored())),
        Variant("sc", "first(1)", (authored(),)),
    )
    second = make_operator(
        "second", "Second", "Beta operators/Second", Variant("ds", "second(DS_1)", (authored(),))
    )
    return Catalogue(
        (
            Chapter("alpha", "Alpha operators", (first,)),
            Chapter("beta", "Beta operators", (second,)),
        )
    )


def test_the_forms_the_names_and_the_notes_are_checked_on_every_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A map with untyped forms, examples named ``minimal`` and no notes: no option asks for the
    # checks and none turns them off, and a run of one chapter, which renders the pages of all of
    # them, checks all of them.
    cat = untyped_catalogue()
    world = build_world(tmp_path, monkeypatch, cat)
    examples = world.root / "examples"
    for key, _ in examples_of(cat):
        (examples / key / "notes.rst").unlink()
    expected = [
        "alpha.first.ds: the form does not declare takes",
        "alpha.first.ds: the form does not declare gives",
        "alpha.first.ds: example name 'minimal' does not say what the example tests",
        "alpha.first.sc: the form does not declare takes",
        "alpha.first.sc: the form does not declare gives",
        "alpha.first.sc: example name 'minimal' does not say what the example tests",
        "beta.second.ds: the form does not declare takes",
        "beta.second.ds: the form does not declare gives",
        "beta.second.ds: example name 'minimal' does not say what the example tests",
        *(f"{examples / key}: missing notes.rst" for key, _ in examples_of(cat)),
    ]
    for only in ([], ["--only", "alpha"], ["--only", "beta"]):
        assert generate.main(["--check-only", *only]) == 1
        assert problems(capsys) == expected
    # A run stops before it starts, and writes nothing. With ``--only`` it asks for the results
    # of the chapters that it leaves out too, after these.
    assert generate.main([]) == 1
    assert problems(capsys) == expected
    assert generate.main(["--only", "beta"]) == 1
    assert problems(capsys)[: len(expected)] == expected
    assert world.calls == []
    assert world.preflights == []
    assert world.names() == ["examples"]


def test_the_notes_of_every_chapter_are_checked(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = world.root / "examples"
    assert generate.main(["--check-only"]) == 0
    capsys.readouterr()
    # In both chapters: notes.md, a missing notes.rst, a notes.rst that is not in the layout.
    (examples / "alpha/first/ds/two_rows/notes.md").write_text("Old notes.\n", encoding="utf-8")
    (examples / "alpha/first/sc/one_value/notes.rst").unlink()
    (examples / "beta/second/ds/two_rows/notes.rst").write_text(":Tests: T.\n", encoding="utf-8")
    expected = [
        f"{examples / 'alpha/first/ds/two_rows/notes.md'}: the notes go in notes.rst, "
        "not in notes.md",
        f"{examples / 'alpha/first/sc/one_value'}: missing notes.rst",
        f"{examples / 'beta/second/ds/two_rows/notes.rst'}: the fields are Tests: {LAYOUT}",
    ]
    for only in ([], ["--only", "alpha"], ["--only", "beta"]):
        assert generate.main(["--check-only", *only]) == 1
        assert problems(capsys) == expected
    assert world.calls == []


# --- The anchors of the forms and the examples of an operator ---


def test_two_examples_that_share_an_anchor_stop_every_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The anchors write the underscores of the ids as dashes: the example ``c`` of the form
    # ``a_b`` and the example ``b_c`` of the form ``a`` are both ``ex-a-b-c``. A duplicate HTML id
    # sends the click of a diagram, the links of a block and of the index and the back link of the
    # page of an example to the wrong example.
    clash = make_operator(
        "clash",
        "Clash",
        "Alpha operators/Clash",
        form("a_b", "clash(a_b)", authored("c")),
        form("a", "clash(a)", authored("b_c")),
    )
    cat = Catalogue((Chapter("alpha", "Alpha operators", (*ALPHA.operators, clash)), BETA))
    world = build_world(tmp_path, monkeypatch, cat, pages=(*PAGES, "Alpha operators/Clash"))
    old = seed_outputs(world)
    expected = ["alpha.clash: the anchor 'ex-a-b-c' is shared by example 'a_b/c', example 'a/b_c'"]
    # Every chapter is checked, whatever ``--only`` says, and the check is the one of every run.
    for only in ([], ["--only", "alpha"], ["--only", "beta"]):
        assert generate.main(["--check-only", *only]) == 1
        assert problems(capsys) == expected
    assert generate.main([]) == 1
    assert problems(capsys) == expected
    # Nothing ran, no child started, and no file was written.
    assert world.calls == []
    assert world.preflights == []
    assert_untouched(world, old)


def anchor_lines(*forms: Variant) -> list[str]:
    """The problems about the anchors of an operator ``alpha.first`` that has ``forms``."""
    operator = make_operator("first", "First", "Alpha operators/First", *forms)
    return list(generate._anchor_problems("alpha.first", operator))


def test_every_anchor_that_is_shared_is_one_line_naming_all_that_have_it() -> None:
    three = anchor_lines(
        form("a_b_c", "x", authored("d"), authored("e_f")),
        form("a_b", "y", authored("c_d"), authored("c_e_f")),
        form("a", "z", authored("b_c_d"), authored("only_one")),
    )
    assert three == [
        "alpha.first: the anchor 'ex-a-b-c-d' is shared by "
        "example 'a_b_c/d', example 'a_b/c_d', example 'a/b_c_d'",
        "alpha.first: the anchor 'ex-a-b-c-e-f' is shared by example 'a_b_c/e_f', "
        "example 'a_b/c_e_f'",
    ]


def test_the_anchors_of_the_forms_are_compared_with_those_of_the_examples() -> None:
    # An id with a dash is no id (``validate_catalogue`` says so), and its anchor is the one of
    # the id with an underscore there: the anchors of the forms are in the comparison too.
    assert anchor_lines(form("a_b", "x", authored("c")), form("a-b", "y", authored("d"))) == [
        "alpha.first: the anchor 'form-a-b' is shared by form 'a_b', form 'a-b'"
    ]


def test_an_anchor_is_not_shared_by_a_duplicate_that_is_reported_already() -> None:
    # A form or an example that is declared twice is a duplicate, which ``validate_catalogue``
    # reports, and it is no other form or example that shares its anchor.
    assert anchor_lines(form("a", "x", authored("b")), form("a", "y", authored("b"))) == []
    assert anchor_lines(form("a", "x", authored("b"), authored("b"))) == []


def test_the_same_example_name_in_two_forms_is_no_shared_anchor() -> None:
    # The anchor of an example has the id of its form: ``ex-ds-one-row`` and ``ex-sc-one-row``.
    assert (
        anchor_lines(form("ds", "x", authored("one_row")), form("sc", "y", authored("one_row")))
        == []
    )


def test_the_real_catalogue_shares_no_anchor() -> None:
    cat = load_catalogue()
    shared = [
        line
        for chapter in cat.chapters
        for operator in chapter.operators
        for line in generate._anchor_problems(f"{chapter.id}.{operator.id}", operator)
    ]
    assert shared == []


def remove_the_script(folder: Path) -> None:
    (folder / "script.vtl").unlink()


def malform_the_expected_structure(folder: Path) -> None:
    (folder / "expected/DS_r.json").write_text("{not json", encoding="utf-8")


def remove_the_notes(folder: Path) -> None:
    (folder / "notes.rst").unlink()


def remove_the_folder(folder: Path) -> None:
    shutil.rmtree(folder)


@pytest.mark.parametrize("only", [[], ["--check-only"]], ids=["a run", "check only"])
@pytest.mark.parametrize(
    ("damage", "problem"),
    [
        (remove_the_script, "missing script.vtl"),
        (malform_the_expected_structure, "DS_r.json: invalid JSON"),
        (remove_the_notes, "missing notes.rst"),
        (remove_the_folder, "missing example folder: examples/alpha/first/ds/two_rows"),
    ],
    ids=["no script", "malformed json", "no notes", "no folder"],
)
def test_a_run_of_one_chapter_checks_the_folders_of_the_others(
    world: World,
    capsys: pytest.CaptureFixture[str],
    damage: Callable[[Path], None],
    problem: str,
    only: list[str],
) -> None:
    # ``--only`` limits which examples run, not what is checked: every run renders the pages of
    # every chapter from their folders. A folder of a chapter that it leaves out that is broken
    # is a problem line and exit 1, as one of a chapter that it runs: nothing is rendered from it.
    assert generate.main([]) == 0
    written = world.files()
    world.calls.clear()
    world.preflights.clear()
    capsys.readouterr()
    damage(world.root / "examples/alpha/first/ds/two_rows")
    assert generate.main(["--only", "beta", *only]) == 1
    [line] = problems(capsys)
    assert problem in line
    # Nothing ran, no child started, and every file is as the full run left it.
    assert world.calls == []
    assert world.preflights == []
    assert world.files() == written


def test_check_only_counts_every_example_that_it_checked(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    # ``--only`` limits what runs, and nothing runs here: all the folders were checked.
    assert generate.main(["--check-only", "--only", "beta"]) == 0
    assert capsys.readouterr().out == f"ok: {len(KEYS)} examples checked, nothing was run\n"


def test_the_flag_strict_forms_is_gone(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        generate.main(["--strict-forms"])
    assert stopped.value.code == 2
    assert "unrecognized arguments: --strict-forms" in capsys.readouterr().err
    assert world.calls == []


# --- The jobs ---


def test_jobs_are_every_example_on_both_engines_and_random_runs_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = make_operator(
        "rnd",
        "Random",
        "Alpha operators/Random",
        form("ds", "random(DS_1, 1)", authored("random_draw", check="random")),
    )
    alpha = Chapter("alpha", "Alpha operators", (*ALPHA.operators, extra))
    cat = Catalogue((alpha, BETA))
    world = build_world(tmp_path, monkeypatch, cat, (*PAGES, "Alpha operators/Random"))
    assert generate.main([]) == 0
    base = world.root / "examples"
    [jobs] = world.calls
    assert jobs == [
        (base / folder, engine, 2 if folder == "alpha/rnd/ds/random_draw" else 1)
        for folder in (*KEYS[:3], "alpha/rnd/ds/random_draw", *KEYS[3:])
        for engine in ("pandas", "duckdb")
    ]
    assert world.settings == [(4, 120.0)]


def test_workers_and_timeout_reach_the_runner(world: World) -> None:
    assert generate.main(["--workers", "2", "--timeout", "30"]) == 0
    assert world.settings == [(2, 30.0)]
    assert world.preflights == [30.0]


def test_a_child_that_cannot_import_the_engines_stops_the_run_before_it_starts(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Every run would be a crash, which would roll up to 🔴 everywhere over the pages there are.
    line = (
        "preflight: a child process cannot import vtlengine and its engines (ModuleNotFoundError: "
        "No module named 'vtlengine.AST.Grammar._cpp_parser.vtl_cpp_parser'); run the generator "
        "with a Python that can import vtlengine: the compiled parser must be in src/"
    )

    def cannot_import(timeout_s: float) -> object:
        raise runner.PreflightError(line)

    monkeypatch.setattr(generate, "preflight", cannot_import, raising=False)
    old = seed_outputs(world)
    assert generate.main([]) == 2
    assert problems(capsys) == [line]
    assert world.calls == []
    assert_untouched(world, old)


def test_only_runs_the_jobs_of_the_chapters_named(world: World) -> None:
    assert generate.main([]) == 0
    world.calls.clear()
    assert generate.main(["--only", "beta"]) == 0
    assert world.run_chapters() == {"beta"}
    [jobs] = world.calls
    assert len(jobs) == 2  # one example on two engines


def test_current_date_is_checked_against_the_day_of_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = make_operator(
        "now",
        "Now",
        "Alpha operators/Now",
        Variant(
            "sc",
            "current_date()",
            (authored("date_of_the_run", check="current_date"),),
            takes=(),
            gives=("date",),
        ),
    )
    cat = Catalogue((Chapter("alpha", "Alpha operators", (today,)),))
    world = build_world(tmp_path, monkeypatch, cat, ("Alpha operators/Now",), ())
    assert generate.main([]) == 0
    assert world.status()["examples"]["alpha/now/sc/date_of_the_run"]["status"] == "ok"
    # The same output on another day is not today.
    monkeypatch.setattr(generate, "_today", lambda: "2026-10-24")
    assert generate.main([]) == 0
    assert world.status()["examples"]["alpha/now/sc/date_of_the_run"]["status"] == "partial"


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
    # The block, its status, its operators, its forms, and its forms by status.
    beta = [":doc:`Beta operators <beta/index>`", "🟡", "1", "1", "0", "1", "0", "0", "0"]
    assert world.index_row("<beta/index>") == beta
    assert world.index_row("<alpha/index>") == [
        ":doc:`Alpha operators <alpha/index>`",
        "🟢",
        "1",
        "2",
        "2",
        "0",
        "0",
        "0",
        "0",
    ]
    # A rerun of beta would now give 🟢, and alpha is 🔴: only alpha is run.
    world.modes = {"alpha": "error"}
    world.calls.clear()
    assert generate.main(["--only", "alpha"]) == 0
    assert world.run_chapters() == {"alpha"}
    assert world.index_row("<alpha/index>") == [
        ":doc:`Alpha operators <alpha/index>`",
        "🔴",
        "1",
        "2",
        "0",
        "0",
        "0",
        "2",
        "0",
    ]
    assert world.index_row("<beta/index>") == beta
    examples = world.status()["examples"]
    assert examples["beta/second/ds/two_rows"]["status"] == "partial"
    assert examples["alpha/first/sc/one_value"]["status"] == "missing"
    assert sorted(examples) == sorted(KEYS)
    assert (world.root / "beta/index.rst").is_file()


def test_only_drops_the_results_of_examples_that_left_the_catalogue(world: World) -> None:
    assert generate.main([]) == 0
    document = world.status()
    document["examples"]["beta/second/ds/gone"] = document["examples"]["beta/second/ds/two_rows"]
    (world.root / "status.json").write_text(json.dumps(document), encoding="utf-8")
    assert generate.main(["--only", "alpha"]) == 0
    assert sorted(world.status()["examples"]) == sorted(KEYS)


def test_only_without_a_previous_result_is_a_validation_problem(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate.main(["--only", "alpha"]) == 1
    lines = problems(capsys)
    assert [line.split(":")[0] for line in lines] == ["no result for beta/second/ds/two_rows"]
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


def test_only_naming_every_chapter_does_not_read_the_previous_results(world: World) -> None:
    (world.root / "status.json").write_text("{not json", encoding="utf-8")
    assert generate.main(["--only", "alpha", "--only", "beta"]) == 0
    assert sorted(world.status()["examples"]) == sorted(KEYS)


def test_only_reads_the_previous_results_when_a_chapter_with_examples_is_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gamma = Chapter(
        "gamma",
        "Gamma operators",
        (
            make_operator(
                "third", "Third", "Gamma operators/Third", form("ds", "t", authored("one_row"))
            ),
        ),
    )
    cat = Catalogue((ALPHA, BETA, gamma))
    world = build_world(tmp_path, monkeypatch, cat, (*PAGES, "Gamma operators/Third"))
    (world.root / "status.json").write_text("{not json", encoding="utf-8")
    assert generate.main(["--only", "alpha", "--only", "beta"]) == 1
    [line] = problems(capsys)
    assert line.startswith("status.json: cannot read the previous results")


@pytest.mark.parametrize(("variants", "more"), [(9, None), (10, 1), (14, 5)])
def test_only_prints_ten_no_result_lines_at_most_and_counts_the_rest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    variants: int,
    more: int | None,
) -> None:
    forms = tuple(form(f"v{n}", f"many(DS_{n})", authored("one_row")) for n in range(variants))
    many = make_operator("many", "Many", "Gamma operators/Many", *forms)
    cat = Catalogue((ALPHA, BETA, Chapter("gamma", "Gamma operators", (many,))))
    world = build_world(tmp_path, monkeypatch, cat, (*PAGES, "Gamma operators/Many"))
    assert generate.main(["--only", "alpha"]) == 1
    lines = problems(capsys)
    # Beta and the variants of gamma have no result: 1 + ``variants`` examples.
    expected = ["beta/second/ds/two_rows", *(f"gamma/many/v{n}/one_row" for n in range(variants))]
    assert [line.split(":")[0] for line in lines[:10]] == [
        f"no result for {x}" for x in expected[:10]
    ]
    if more is None:
        assert len(lines) == 10
    else:
        assert lines[10:] == [f"… and {more} more"]
    assert world.calls == []


def test_only_upgrades_the_results_of_a_status_file_of_the_old_format(world: World) -> None:
    world.modes = {"alpha": "refused"}
    assert generate.main([]) == 0
    document = world.status()
    # What the tool before this one wrote: no fields for the error, and no refused status.
    for entry in document["examples"].values():
        for engine in ("pandas", "duckdb"):
            del entry[engine]["error_type"]
            del entry[engine]["error_code"]
        if entry["status"] == "refused":
            entry["status"] = "missing"
    (world.root / "status.json").write_text(json.dumps(document), encoding="utf-8")

    assert generate.main(["--only", "beta"]) == 0

    examples = world.status()["examples"]
    entry = examples["alpha/first/sc/one_value"]
    assert entry["status"] == "refused"  # judged again, with the rules of this tool
    assert entry["pandas"]["error_type"] == "RunTimeError"  # read from the detail
    assert entry["pandas"]["error_code"] == "2-1-15-6"
    assert examples["beta/second/ds/two_rows"]["pandas"]["error_code"] is None


def test_parse_verdict_old_file() -> None:
    raw = {
        "outcome": "error",
        "detail": "RunTimeError 2-1-15-6: At op mod: Scalar division by Zero.",
    }
    got = generate._parse_verdict(raw)
    assert got == EngineVerdict("error", raw["detail"], "RunTimeError", "2-1-15-6")


@pytest.mark.parametrize(
    ("detail", "error_type", "error_code"),
    [
        ("SemanticError 1-2-23: a code of three groups", "SemanticError", "1-2-23"),
        (
            "InputValidationException 0-5-1-9: not implemented",
            "InputValidationException",
            "0-5-1-9",
        ),
        ("DataLoadError 0-3-1-6: cannot load", "DataLoadError", "0-3-1-6"),
        ("AttributeError: 'NoneType' object has no attribute 'data'", "AttributeError", None),
        ("KeyError: 'Me_1'", "KeyError", None),
        ("Timeout: no result in 120 s, 2 attempts", "Timeout", None),
        ("Crash: exit 1: Segmentation fault", "Crash", None),
        ("ParserException: Parser Error: syntax error at or near x", "ParserException", None),
        # A run that said no message has the head alone: its type, or its type and its code.
        ("SemanticError 1-1-1-1", "SemanticError", "1-1-1-1"),
        ("InputValidationException 0-5-1-9", "InputValidationException", "0-5-1-9"),
        ("KeyError", "KeyError", None),
        ("Crash", "Crash", None),
    ],
)
def test_parse_verdict_of_an_old_file_reads_the_error_of_its_detail(
    detail: str, error_type: str, error_code: str | None
) -> None:
    got = generate._parse_verdict({"outcome": "error", "detail": detail})
    assert got == EngineVerdict("error", detail, error_type, error_code)


@pytest.mark.parametrize(
    "detail",
    [
        "no operator",  # a message with no type: a run that failed with only this to say
        "1-1-1-1: no operator",  # a code with no type
        "SemanticError 1-1-1-1 and more",  # not the head of an error: no colon, and not the end
        "",
    ],
)
def test_parse_verdict_of_an_old_file_reads_no_error_from_a_message_that_has_no_type(
    detail: str,
) -> None:
    got = generate._parse_verdict({"outcome": "error", "detail": detail})
    assert got == EngineVerdict("error", detail, None, None)


@pytest.mark.parametrize(
    ("outcome", "detail"),
    [
        ("match", ""),
        # Each of these starts like ``<type>:``, and none of them is an error.
        ("mismatch", "DS_r: row Id_1=1: Me_1 expected 3 got 4"),
        ("mismatch", "sc_r: type expected Number got Integer"),
        ("mismatch", "missing output DS_r"),
        ("mismatch", "expected error 2-1-15-6, got 2-1-3-1"),
    ],
)
def test_parse_verdict_of_an_old_file_reads_no_error_from_what_is_not_one(
    outcome: str, detail: str
) -> None:
    got = generate._parse_verdict({"outcome": outcome, "detail": detail})
    assert (got.outcome, got.detail) == (outcome, detail)
    assert (got.error_type, got.error_code) == (None, None)


def test_parse_verdict_reads_the_fields_when_the_file_has_them() -> None:
    raw: dict[str, Any] = {
        "outcome": "error",
        "detail": "d",
        "error_type": "SemanticError",
        "error_code": "1-1-1-4",
    }
    assert generate._parse_verdict(raw) == EngineVerdict("error", "d", "SemanticError", "1-1-1-4")
    # What is stored is what is read, even where the detail says something else.
    raw = {
        "outcome": "error",
        "detail": "RunTimeError 2-1-15-6: x",
        "error_type": "KeyError",
        "error_code": None,
    }
    assert generate._parse_verdict(raw) == EngineVerdict(
        "error", "RunTimeError 2-1-15-6: x", "KeyError", None
    )
    # A mismatch of an example that expects an error has the error that the run raised.
    raw = {
        "outcome": "mismatch",
        "detail": "expected error 2-1-15-6, got 2-1-3-1",
        "error_type": "RunTimeError",
        "error_code": "2-1-3-1",
    }
    assert generate._parse_verdict(raw) == EngineVerdict(
        "mismatch", "expected error 2-1-15-6, got 2-1-3-1", "RunTimeError", "2-1-3-1"
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"error_type": 3, "error_code": None},
        {"error_type": "SemanticError", "error_code": 1},
        {"error_type": ["SemanticError"], "error_code": "1-1-1-1"},
    ],
)
def test_parse_verdict_rejects_error_fields_that_are_not_texts(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="not a verdict"):
        generate._parse_verdict({"outcome": "error", "detail": "d", **fields})


def stored(
    status: str, pandas: dict[str, str], duckdb: dict[str, str], agree: bool = True
) -> dict[str, Any]:
    """The entry of an example in a ``status.json`` of the old format: no error fields."""
    return {"status": status, "engines_agree": agree, "pandas": pandas, "duckdb": duckdb}


def test_a_stored_status_is_judged_again_with_the_rules_of_the_tool(tmp_path: Path) -> None:
    refused = {"outcome": "error", "detail": "RunTimeError 2-1-15-6: Scalar division by Zero."}
    gap = {"outcome": "error", "detail": "SemanticError 1-3-13: Unsupported VTL construct: eval"}
    crash = {"outcome": "error", "detail": "AttributeError: 'NoneType' object has no attribute 'a'"}
    bare_refused = {"outcome": "error", "detail": "RunTimeError 2-1-15-6"}  # it said no message
    bare_crash = {"outcome": "error", "detail": "KeyError"}
    matched = {"outcome": "match", "detail": ""}
    wrong = {"outcome": "mismatch", "detail": "DS_r: row Id_1=1: Me_1 expected 3 got 4"}
    entries = {
        # The tool that wrote these called each of them what it knew: ok, missing or partial.
        "refused": stored("missing", refused, refused),
        "not_implemented": stored("missing", gap, gap),
        "crash": stored("missing", crash, crash),
        "refused_and_gap": stored("missing", refused, gap),
        "refused_and_crash": stored("missing", refused, crash),
        "bare_refused": stored("missing", bare_refused, bare_refused),
        "bare_crash": stored("missing", bare_crash, bare_crash),
        "match": stored("ok", matched, matched),
        "wrong": stored("partial", wrong, wrong),
        "diverge": stored("diverge", matched, refused, agree=False),
        # A status that the file holds is not trusted: it is always worked out again.
        "stale": stored("ok", refused, refused),
    }
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"meta": {}, "examples": entries}), encoding="utf-8")

    previous = generate._load_previous(path)

    assert {key: result.status for key, result in previous.items()} == {
        "refused": "refused",
        "not_implemented": "missing",
        "crash": "missing",
        "refused_and_gap": "missing",
        "refused_and_crash": "missing",
        "bare_refused": "refused",
        "bare_crash": "missing",
        "match": "ok",
        "wrong": "partial",
        "diverge": "diverge",
        "stale": "refused",
    }
    assert previous["refused"].pandas == EngineVerdict(
        "error", refused["detail"], "RunTimeError", "2-1-15-6"
    )
    assert previous["crash"].duckdb.error_type == "AttributeError"
    assert previous["crash"].duckdb.error_code is None
    assert previous["bare_refused"].pandas == EngineVerdict(
        "error", "RunTimeError 2-1-15-6", "RunTimeError", "2-1-15-6"
    )
    assert previous["bare_crash"].pandas == EngineVerdict("error", "KeyError", "KeyError", None)


def test_a_stored_status_that_is_not_a_status_is_rejected(tmp_path: Path) -> None:
    entry = stored("green", {"outcome": "match", "detail": ""}, {"outcome": "match", "detail": ""})
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"meta": {}, "examples": {"a": entry}}), encoding="utf-8")
    with pytest.raises(generate.GenerateError, match="cannot read the previous results"):
        generate._load_previous(path)


# --- What is written ---


def test_the_run_writes_every_file_and_leaves_nothing_else(world: World) -> None:
    assert generate.main([]) == 0
    # Sphinx pages in a folder for each block, the results and the manifest: no flat page for a
    # block, and no Markdown file.
    assert world.tree() == sorted([MANIFEST, "alpha/", "beta/", *OUTPUTS])
    index = (world.root / "index.rst").read_text(encoding="utf-8")
    rule = "=" * 40
    assert index.startswith(f"{rule}\nVTL 2.2 operator map\n{rule}\n")
    # The index has the parts of the map in its toctree, the overview has the page of the
    # operators (the map of the world has no definitions), and that page has the blocks.
    assert (
        ".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n   how_to_read\n   overview\n   decisions\n"
    ) in index
    overview = (world.root / "overview.rst").read_text(encoding="utf-8")
    assert overview.endswith(".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n   operators\n")
    operators = (world.root / "operators.rst").read_text(encoding="utf-8")
    assert (
        ".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n   alpha/index\n   beta/index\n" in operators
    )
    assert "9.9.9" in index
    assert "abc1234" in index
    assert f"with pandas {PANDAS_VERSION} and DuckDB {DUCKDB_VERSION}" in index
    assert f"(sdmx-twg/vtl @ {SPEC_COMMIT})" in overview  # the attribution
    alpha = (world.root / "alpha/index.rst").read_text(encoding="utf-8")
    rule = "=" * 30  # twice the 15 characters of "Alpha operators"
    assert alpha.startswith(f"{rule}\nAlpha operators\n{rule}\n")
    assert "🟢 1 of 1 operator all ok" in alpha
    assert ".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n   first\n" in alpha
    assert (
        '.. mermaid::\n   :config: {"flowchart": {"useMaxWidth": false}}\n\n   flowchart LR\n'
        in (alpha)
    )


def test_nested_write_creates_dirs(world: World) -> None:
    # The folder has no folder of a block: the run makes them, and the pages in them are the ones
    # that the hidden toctrees name, each relative to the page that holds it.
    assert not (world.root / "alpha").exists()
    assert generate.main([]) == 0
    for name in OUTPUTS:
        assert (world.root / name).is_file(), name
    block = (world.root / "alpha/index.rst").read_text(encoding="utf-8")
    operator = (world.root / "alpha/first.rst").read_text(encoding="utf-8")
    assert "   first\n" in block
    title = "First · ``first``"  # the name, and the symbol, which a test operator has as its id
    rule = "=" * (2 * len(title))
    assert operator.startswith(f"{rule}\n{title}\n{rule}\n")
    # The page of an operator links to the files of its examples relative to its own folder, and
    # every link is to a file that is there.
    scripts = re.findall(r"^\.\. literalinclude:: (\S+)$", operator, flags=re.MULTILINE)
    assert sorted(scripts) == [
        "../examples/alpha/first/ds/manual_ex_1/script.vtl",
        "../examples/alpha/first/ds/two_rows/script.vtl",
        "../examples/alpha/first/sc/one_value/script.vtl",
    ]
    assert all((world.root / "alpha" / script).resolve().is_file() for script in scripts)
    # The second run finds the folders there, and leaves them as they are.
    assert generate.main([]) == 0
    assert world.tree() == sorted([MANIFEST, "alpha/", "beta/", *OUTPUTS])


def test_the_manifest_lists_every_generated_path_sorted(world: World) -> None:
    assert generate.main([]) == 0
    text = (world.root / MANIFEST).read_text(encoding="utf-8")
    # One path per line, sorted, with the folders of the blocks, and not the manifest itself.
    assert text.splitlines() == sorted(OUTPUTS)
    assert text.endswith("\n")
    assert MANIFEST not in text.splitlines()
    # A run that leaves out a chapter writes the same pages and the same manifest.
    assert generate.main(["--only", "beta"]) == 0
    assert (world.root / MANIFEST).read_text(encoding="utf-8") == text
    assert all((world.root / name).is_file() for name in OUTPUTS)


def test_stale_file_removed(world: World) -> None:
    old = seed_outputs(world)
    # A page that a person wrote in a folder of the map is not in the manifest: it stays.
    hand_written = world.root / "alpha/hand_written.rst"
    hand_written.write_text("Mine.\n", encoding="utf-8")
    assert generate.main([]) == 0
    # The pages of the manifest that this run does not generate are gone, and the folder that
    # they were alone in is gone too; the folder of a block that is still there is not.
    for name in STALE:
        assert not (world.root / name).exists(), name
    assert not (world.root / "gone").exists()
    assert hand_written.read_text(encoding="utf-8") == "Mine.\n"
    assert world.tree() == sorted([MANIFEST, "alpha/", "alpha/hand_written.rst", "beta/", *OUTPUTS])
    # Every file that this run generates is new, and the manifest no longer lists the stale ones.
    assert all((world.root / name).read_bytes() != old[name] for name in OUTPUTS)
    manifest = (world.root / MANIFEST).read_text(encoding="utf-8")
    assert manifest.splitlines() == sorted(OUTPUTS)
    # The examples of the map are never touched.
    assert (world.root / "examples/alpha/first/ds/two_rows/script.vtl").is_file()


def test_the_manifest_is_moved_last_after_the_removals(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A run that is killed half way (no rollback) leaves the old manifest, which still lists the
    # files that the next run is to remove: so the files come first, then the removals, and the
    # manifest last.
    old = seed_outputs(world)
    events: list[tuple[str, str]] = []
    real_replace, real_unlink = os.replace, Path.unlink

    def replace(source: Any, target: Any) -> None:
        place = staged(source)
        if place is not None and place[0] == "new":
            events.append(("move", Path(target).relative_to(world.root).as_posix()))
        real_replace(source, target)

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.relative_to(world.root).as_posix() in STALE:
            events.append(("remove", self.relative_to(world.root).as_posix()))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    assert generate.main([]) == 0
    assert events == [
        *(("move", name) for name in OUTPUTS),
        *(("remove", name) for name in STALE),
        ("move", MANIFEST),
    ]
    assert (world.root / MANIFEST).read_bytes() != old[MANIFEST]


def test_a_renamed_operator_loses_its_old_page(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert generate.main([]) == 0
    assert (world.root / "alpha/first.rst").is_file()
    # ``first`` becomes ``renamed``: its examples move with it.
    renamed = make_operator(
        "renamed",
        "Renamed",
        "Alpha operators/First",
        form("ds", "first(DS_1)", manual("ex_1"), authored("two_rows")),
        form("sc", "first(1)", authored("one_value")),
    )
    cat = Catalogue((Chapter("alpha", "Alpha operators", (renamed,)), BETA))
    shutil.rmtree(world.root / "examples/alpha/first")
    write_examples(world.root, Catalogue((Chapter("alpha", "Alpha operators", (renamed,)),)))
    monkeypatch.setattr(generate, "load_catalogue", lambda: cat)
    assert generate.main([]) == 0
    assert not (world.root / "alpha/first.rst").exists()
    assert (world.root / "alpha/renamed.rst").is_file()
    assert "   renamed\n" in (world.root / "alpha/index.rst").read_text(encoding="utf-8")
    manifest = (world.root / MANIFEST).read_text(encoding="utf-8").splitlines()
    assert "alpha/first.rst" not in manifest
    assert "alpha/renamed.rst" in manifest
    assert manifest == sorted(manifest)
    # A chapter that is gone takes its folder with it.
    monkeypatch.setattr(generate, "load_catalogue", lambda: Catalogue((ALPHA,)))
    shutil.rmtree(world.root / "examples/alpha/renamed")
    shutil.rmtree(world.root / "examples/beta")
    write_examples(world.root, Catalogue((ALPHA,)))
    monkeypatch.setattr(manual_index, "MANUAL_PAGES", PAGES[:1])
    assert generate.main([]) == 0
    assert not (world.root / "beta").exists()
    assert not (world.root / "alpha/renamed.rst").exists()
    assert (world.root / "alpha/first.rst").is_file()


@pytest.mark.parametrize(
    "line",
    [
        "../outside.rst",
        "alpha/../../outside.rst",
        "/etc/passwd",
        "examples/alpha/first/ds/two_rows/script.vtl",
        "alpha//first.rst",
        "./first.rst",
        ".generated",
        "alpha\\first.rst",
        "",
        "generate.py",
        "catalogue/numeric.py",
        "tool/render.rst",
        "alpha/deep/first.rst",
        "Alpha/first.rst",
        "alpha/status.json",
    ],
    ids=[
        "up",
        "up inside",
        "absolute",
        "example",
        "empty part",
        "dot",
        "manifest",
        "backslash",
        "empty",
        "source",
        "catalogue",
        "kept folder",
        "too deep",
        "upper case",
        "json in a block",
    ],
)
def test_a_manifest_that_names_a_path_outside_the_pages_is_a_problem(
    world: World, capsys: pytest.CaptureFixture[str], tmp_path: Path, line: str
) -> None:
    # A manifest that was edited, or cut short, must never make a run delete what it did not
    # write: nothing is run, nothing is removed, and the problem says which line.
    outside = tmp_path / "outside.rst"
    outside.write_text("Not mine.\n", encoding="utf-8")
    old = seed_outputs(world)
    manifest = (world.root / MANIFEST).read_text(encoding="utf-8") + f"{line}\n"
    (world.root / MANIFEST).write_text(manifest, encoding="utf-8")
    old[MANIFEST] = manifest.encode()
    assert generate.main([]) == 1
    [message] = problems(capsys)
    number = len(manifest.splitlines())
    assert message.startswith(f"{MANIFEST}: line {number} is not the path of a generated file")
    assert world.calls == []
    assert outside.read_text(encoding="utf-8") == "Not mine.\n"
    assert_untouched(world, old)


def test_a_manifest_line_through_a_symbolic_link_is_a_problem(
    world: World, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # A folder of the map that links elsewhere: the line has the shape of a page, but its file is
    # outside the pages, so nothing is run and nothing is removed.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.rst").write_text("Not mine.\n", encoding="utf-8")
    seed_outputs(world)
    (world.root / "linked").symlink_to(outside, target_is_directory=True)
    manifest = (world.root / MANIFEST).read_text(encoding="utf-8") + "linked/victim.rst\n"
    (world.root / MANIFEST).write_text(manifest, encoding="utf-8")
    assert generate.main([]) == 1
    [message] = problems(capsys)
    number = len(manifest.splitlines())
    assert message.startswith(f"{MANIFEST}: line {number} goes through a symbolic link")
    assert world.calls == []
    assert (outside / "victim.rst").read_text(encoding="utf-8") == "Not mine.\n"
    assert (world.root / MANIFEST).read_text(encoding="utf-8") == manifest


def test_a_page_name_that_another_file_needs_is_a_problem(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # An operator called ``index`` would overwrite the page of its block, and a chapter called
    # ``tool`` would write its pages into the tool: neither runs.
    index_op = make_operator(
        "index", "Index", "Alpha operators/First", form("ds", "first(DS_1)", authored("one_row"))
    )
    cat = Catalogue(
        (
            Chapter("alpha", "Alpha operators", (index_op,)),
            Chapter("tool", "Beta operators", BETA.operators),
        )
    )
    monkeypatch.setattr(generate, "load_catalogue", lambda: cat)
    assert generate.main([]) == 1
    found = problems(capsys)
    assert "alpha.index: its page would be the page of the block, alpha/index.rst" in found
    assert "tool: its pages would go in the folder tool/ of the map" in found
    assert world.calls == []


def test_only_removes_the_stale_files_and_keeps_the_other_pages(world: World) -> None:
    # A run of one chapter renders every page: the pages of the chapter it leaves out stay as the
    # full run wrote them and the manifest still lists them, while the stale files are removed.
    assert generate.main([]) == 0
    alpha = {name: (world.root / name).read_bytes() for name in OUTPUTS if name[:6] == "alpha/"}
    for name in STALE:
        (world.root / name).parent.mkdir(parents=True, exist_ok=True)
        (world.root / name).write_text("Stale.\n", encoding="utf-8")
    lines = sorted([*OUTPUTS, *STALE])
    (world.root / MANIFEST).write_text("".join(f"{name}\n" for name in lines), encoding="utf-8")
    assert generate.main(["--only", "beta"]) == 0
    assert not any((world.root / name).exists() for name in STALE)
    assert not (world.root / "gone").exists()
    assert {name: (world.root / name).read_bytes() for name in alpha} == alpha
    assert (world.root / MANIFEST).read_text(encoding="utf-8").splitlines() == sorted(OUTPUTS)


def test_a_manifest_that_cannot_be_read_is_a_problem(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    (world.root / MANIFEST).write_bytes(b"alpha/first.rst\n\xff\xfe\n")
    assert generate.main([]) == 1
    [message] = problems(capsys)
    assert message.startswith(f"{MANIFEST}: cannot read the files of the previous run")
    assert world.calls == []


def test_a_stale_path_that_is_gone_or_is_a_folder_is_left_alone(world: World) -> None:
    # The manifest names a file that someone removed, and a folder with the name of a page:
    # neither is an error, and a folder is never removed, with what it holds.
    (world.root / "alpha/kept.rst").mkdir(parents=True)
    (world.root / "alpha/kept.rst/note.txt").write_text("Kept.\n", encoding="utf-8")
    (world.root / MANIFEST).write_text("alpha/gone.rst\nalpha/kept.rst\n", encoding="utf-8")
    assert generate.main([]) == 0
    assert (world.root / "alpha/kept.rst/note.txt").read_text(encoding="utf-8") == "Kept.\n"
    assert not (world.root / "alpha/gone.rst").exists()


def test_example_pages_json(world: World) -> None:
    world.modes = {"alpha": "diverge"}
    assert generate.main([]) == 0
    text = (world.root / "example_pages.json").read_text(encoding="utf-8")
    pages = json.loads(text)
    # Sorted keys and no escaped emoji, as ``status.json`` is written, and a line feed at the end.
    assert text == json.dumps(pages, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert sorted(pages) == sorted(KEYS)
    assert pages["alpha/first/ds/manual_ex_1"] == {
        "title": "Manual example 1",
        "status": "diverge",
        "pandas": "✅",
        "duckdb": "🛡 1-1-1-1",
        "operator_page": "alpha/first.html",
        "anchor": "ex-ds-manual-ex-1",
    }
    assert pages["alpha/first/sc/one_value"] == {
        "title": "One value",
        "status": "diverge",
        "pandas": "✅",
        "duckdb": "🛡 1-1-1-1",
        "operator_page": "alpha/first.html",
        "anchor": "ex-sc-one-value",
    }
    assert pages["beta/second/ds/two_rows"] == {
        "title": "Two rows",
        "status": "ok",
        "pandas": "✅",
        "duckdb": "✅",
        "operator_page": "beta/second.html",
        "anchor": "ex-ds-two-rows",
    }
    # Every entry is the example of its key, on the page that it names, under the anchor that it
    # names: the page of the operator has the example as an id, and the page is a file.
    for key, entry in pages.items():
        assert set(entry) == {"title", "status", "pandas", "duckdb", "operator_page", "anchor"}
        chapter, operator, variant, example = key.split("/")
        assert entry["operator_page"] == f"{chapter}/{operator}.html"
        page = (world.root / entry["operator_page"].removesuffix(".html")).with_suffix(".rst")
        assert f'id="{entry["anchor"]}"' in page.read_text(encoding="utf-8"), key
        assert entry["anchor"] == f"ex-{variant}-{example}".replace("_", "-")


def test_example_pages_json_marks_guarded_errors_and_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path, monkeypatch)

    def outputs(_self: World, _path: Path, engine: Engine) -> RunOutput:
        if engine == "pandas":
            return RunOutput(False, [], [], "AttributeError", None, "no attribute on pandas")
        return RunOutput(False, [], [], "RunTimeError", "2-1-15-6", "refused on duckdb")

    monkeypatch.setattr(World, "output", outputs)
    assert generate.main([]) == 0
    pages = json.loads((world.root / "example_pages.json").read_text(encoding="utf-8"))
    entry = pages["beta/second/ds/two_rows"]
    assert (entry["pandas"], entry["duckdb"]) == ("💥 AttributeError", "🛡 2-1-15-6")
    assert entry["status"] == "missing"


def test_status_json_has_sorted_keys_the_meta_and_every_verdict(world: World) -> None:
    world.modes = {"alpha": "diverge"}
    assert generate.main([]) == 0
    text = (world.root / "status.json").read_text(encoding="utf-8")
    document = json.loads(text)
    assert text == json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert document["meta"] == {
        "date": TODAY,
        "duckdb_version": DUCKDB_VERSION,
        "pandas_version": PANDAS_VERSION,
        "spec_commit": SPEC_COMMIT,
        "vtlengine_commit": "abc1234",
        "vtlengine_version": "9.9.9",
    }
    assert sorted(document["examples"]) == sorted(KEYS)
    assert document["examples"]["alpha/first/sc/one_value"] == {
        "duckdb": {
            "detail": "SemanticError 1-1-1-1: failed on duckdb",
            "error_code": "1-1-1-1",
            "error_type": "SemanticError",
            "outcome": "error",
        },
        "engines_agree": False,
        "pandas": {"detail": "", "error_code": None, "error_type": None, "outcome": "match"},
        "status": "diverge",
    }
    assert document["examples"]["beta/second/ds/two_rows"]["status"] == "ok"


def test_a_refused_example_is_written_and_read_back(world: World) -> None:
    world.modes = {"alpha": "refused"}
    assert generate.main([]) == 0
    entry = world.status()["examples"]["alpha/first/sc/one_value"]
    assert entry["status"] == "refused"
    assert entry["engines_agree"] is True
    for engine in ("pandas", "duckdb"):
        assert entry[engine] == {
            "detail": f"RunTimeError 2-1-15-6: refused on {engine}",
            "error_code": "2-1-15-6",
            "error_type": "RunTimeError",
            "outcome": "error",
        }
    # Only beta runs now: the results of alpha come back from the file as they were written.
    world.calls.clear()
    assert generate.main(["--only", "beta"]) == 0
    assert world.run_chapters() == {"beta"}
    assert world.status()["examples"]["alpha/first/sc/one_value"] == entry
    assert world.status()["examples"]["alpha/first/ds/two_rows"]["status"] == "refused"


def test_a_crash_is_written_with_its_type_and_no_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path, monkeypatch)

    def crash(_self: World, _path: Path, engine: Engine) -> RunOutput:
        return RunOutput(False, [], [], "AttributeError", None, f"no attribute on {engine}")

    monkeypatch.setattr(World, "output", crash)
    assert generate.main([]) == 0
    document = world.status()
    entry = document["examples"]["alpha/first/sc/one_value"]
    assert entry["status"] == "missing"
    assert entry["pandas"]["error_type"] == "AttributeError"
    assert entry["pandas"]["error_code"] is None
    assert {each["status"] for each in document["examples"].values()} == {"missing"}


def test_a_rerun_gives_the_same_files(world: World) -> None:
    assert generate.main([]) == 0
    first = world.files()
    assert generate.main([]) == 0
    assert world.files() == first
    assert sorted(first) == sorted([*OUTPUTS, MANIFEST])


def test_the_summary_says_the_stale_files_and_the_example_statuses_apart(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    seed_outputs(world)  # three files that no run generates: they are the stale ones
    world.modes = {"beta": "error"}  # one example is 🔴, and the other three are 🟢
    assert generate.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"running {len(KEYS)} examples on both engines (4 at a time)"
    # The number of files that were removed, then the statuses of the examples, each as a
    # statement of its own: the counts of the statuses are not read as counts of stale files.
    assert lines[1:] == [
        "wrote 9 pages, status.json, example_pages.json and .generated; removed 3 stale files. "
        "Examples: 🟢 3 · 🟡 0 · 🛡️ 0 · 🔴 1 · 🔵 0"
    ]


def test_a_block_over_a_limit_exits_1_and_writes_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def too_big(*_args: object, **_kwargs: object) -> str:
        raise MermaidLimitError("overview: 501 edges, and a block needs fewer than 500")

    monkeypatch.setattr(generate, "index_rst", too_big)
    old = seed_outputs(world)
    assert generate.main([]) == 1
    assert problems(capsys) == ["overview: 501 edges, and a block needs fewer than 500"]
    assert_untouched(world, old)


def test_every_block_of_the_rendered_files_is_checked_against_the_limits(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    edges = "\n".join(f"       a{n} --> b{n}" for n in range(500))
    big = f"Chapter\n=======\n\n.. mermaid::\n\n   flowchart LR\n{edges}\n\ntext\n"
    monkeypatch.setattr(generate, "block_rst", lambda *_args, **_kwargs: big)
    old = seed_outputs(world)
    assert generate.main([]) == 1
    [line] = problems(capsys)
    assert line.startswith("alpha/index.rst: ")
    assert "500 edges" in line
    assert_untouched(world, old)


def test_a_diagram_of_an_operator_page_is_checked_against_the_limits_too(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    edges = "\n".join(f"       a{n} --> b{n}" for n in range(500))
    big = f"Operator\n========\n\n.. mermaid::\n\n   flowchart LR\n{edges}\n\ntext\n"
    monkeypatch.setattr(generate, "operator_rst", lambda *_args, **_kwargs: big)
    old = seed_outputs(world)
    assert generate.main([]) == 1
    [line] = problems(capsys)
    assert line.startswith("alpha/first.rst: ")
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
    monkeypatch.setattr(generate, "block_rst", lambda *_a, **_k: "lone \udc80 surrogate\n")


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
    monkeypatch.setattr(generate, "block_rst", lambda *_a, **_k: interrupt())


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
        interrupt_the_replace(len(OUTPUTS) + 1),
        interrupt_the_replace(len(OUTPUTS) + 1, after=True),
    ],
    ids=[
        "judging",
        "rendering",
        "first replace",
        "third replace",
        "right after the first replace",
        "right after the third replace",
        "manifest",
        "right after the manifest",
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


def interrupt_the_unlink(
    name: str, *, after: bool = False
) -> Callable[[World, pytest.MonkeyPatch], None]:
    """The removal of the stale file ``name`` is interrupted: before it, or right after it."""

    def inject(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
        real_unlink = Path.unlink
        target = world.root / name

        def flaky(self: Path, missing_ok: bool = False) -> None:
            if self == target and not after:
                interrupt()
            real_unlink(self, missing_ok=missing_ok)
            if self == target:
                interrupt()

        monkeypatch.setattr(Path, "unlink", flaky)

    return inject


@pytest.mark.parametrize(
    "inject",
    [
        interrupt_the_unlink(STALE[0]),
        interrupt_the_unlink(STALE[0], after=True),
        interrupt_the_unlink(STALE[-1]),
        interrupt_the_unlink(STALE[-1], after=True),
        interrupt_the_replace(len(OUTPUTS) + 1),
        interrupt_the_replace(len(OUTPUTS) + 1, after=True),
    ],
    ids=[
        "first stale file",
        "right after the first stale file",
        "last stale file",
        "right after the last stale file",
        "manifest",
        "right after the manifest",
    ],
)
def test_interrupt_keeps_old_tree(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    inject: Callable[[World, pytest.MonkeyPatch], None],
) -> None:
    # The old tree has stale files in a folder that stays and in one that goes. An interrupt in
    # the removal of any of them, or in the move of the manifest after them, leaves every file,
    # the old manifest and the stale files as they were, and no folder is made or removed.
    old = seed_outputs(world)
    inject(world, monkeypatch)
    assert run_main_interrupted() == 2
    assert capsys.readouterr().err.strip() == "interrupted: nothing was written"
    assert_untouched(world, old)


def staging_folders(world: World) -> list[Path]:
    """The staging folders of the generator that are left in its folder."""
    return [path for path in world.root.iterdir() if path.name.startswith(".generate-")]


def failure(kind: type[BaseException]) -> BaseException:
    """What a stubbed ``os.replace`` raises: a failure of the system, or an interrupt."""
    return OSError(errno.EACCES, "no way") if kind is OSError else KeyboardInterrupt()


# The move of a new file that fails (the nth), and the file that cannot be put back after it: a
# file of the first block, at the top level and in the folder of a block; and, when the move of
# the manifest fails after the stale files were removed, a stale file in a folder that was going
# to be removed. The third is the names that the run noted for putting back.
RESTORE_CASES = [
    (3, "index.rst", OUTPUTS[:3]),
    (7, "alpha/index.rst", OUTPUTS[:7]),
    (len(OUTPUTS) + 1, "gone/page.rst", [*OUTPUTS, *STALE, MANIFEST]),
]


@pytest.mark.parametrize(
    "restore_fails", [OSError, KeyboardInterrupt], ids=["failed restore", "second interrupt"]
)
@pytest.mark.parametrize(
    "replace_fails", [OSError, KeyboardInterrupt], ids=["failed replace", "interrupted replace"]
)
@pytest.mark.parametrize(
    ("nth", "failed", "touched"),
    RESTORE_CASES,
    ids=["top level", "in a folder", "stale file"],
)
def test_restore_with_subdirs(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    nth: int,
    failed: str,
    touched: list[str],
    replace_fails: type[BaseException],
    restore_fails: type[BaseException],
) -> None:
    old = seed_outputs(world)
    real_replace = os.replace
    calls: list[tuple[str, str]] = []  # (new or old, the name of the file) of every move

    def flaky(source: Any, target: Any) -> None:
        place = staged(source)
        if place is None:
            real_replace(source, target)
            return
        calls.append(place)
        moves = sum(1 for where, _ in calls if where == "new")
        # The nth file that is moved cannot be moved, and one file cannot be put back, for a
        # failure of the system or because the run is interrupted again.
        if place[0] == "new" and moves == nth:
            raise failure(replace_fails)
        if place == ("old", failed):
            raise failure(restore_fails)
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", flaky)
    assert generate.main([]) == 2
    [line] = problems(capsys)
    # Every file that the run noted, the one that could not be moved too, and every stale file
    # that it had removed, was tried, whatever happened to the ones before it.
    assert sorted(name for where, name in calls if where == "old") == sorted(touched)
    # The staging folder has the old bytes of the file that could not be put back, in the folders
    # of its name. The run left it as it was (the new bytes, or none for a stale file that it
    # had removed), and every other file is as it was.
    [staging] = staging_folders(world)
    assert (staging / "old" / failed).read_bytes() == old[failed]
    if failed in STALE:
        assert not (world.root / failed).exists()
    else:
        assert (world.root / failed).read_bytes() != old[failed]
    others = {name: data for name, data in old.items() if name != failed}
    assert {name: (world.root / name).read_bytes() for name in others} == others
    assert line.startswith("internal error: ")
    assert str(staging) in line
    assert failed in line


def new_files_are_in_place(world: World, old: dict[str, bytes]) -> bool:
    """Whether every output file is a new one, the manifest too, and the stale files are gone."""
    written = all((world.root / name).read_bytes() != old[name] for name in [*OUTPUTS, MANIFEST])
    return written and not any((world.root / name).exists() for name in STALE)


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


@pytest.mark.parametrize("where", ["help", "docstring"])
def test_the_help_and_the_docstring_say_how_to_run_the_generator(
    where: str, capsys: pytest.CaptureFixture[str]
) -> None:
    if where == "help":
        with pytest.raises(SystemExit) as stopped:
            generate.main(["--help"])
        assert stopped.value.code == 0
        text = capsys.readouterr().out
    else:
        text = generate.__doc__ or ""
    words = " ".join(text.split())
    # A Python that can import vtlengine with its compiled parser, under a memory cap where
    # systemd can set one, and without one elsewhere.
    assert "a Python that can import vtlengine, with the compiled parser in src/" in words
    assert (
        "systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 python "
        "docs/operator_map/generate.py"
    ) in words
    assert "Elsewhere the cap is optional." in words


def test_the_help_names_every_file_that_a_run_writes_besides_the_pages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        generate.main(["--help"])
    words = " ".join(capsys.readouterr().out.split())
    assert f"{generate._STATUS_FILE} and {generate._EXAMPLE_PAGES_FILE}." in words


@pytest.mark.parametrize("where", ["help", "docstring"])
def test_the_help_and_the_docstring_say_that_every_chapter_is_checked(
    where: str, capsys: pytest.CaptureFixture[str]
) -> None:
    if where == "help":
        with pytest.raises(SystemExit):
            generate.main(["--help"])
        text = capsys.readouterr().out
    else:
        text = generate.__doc__ or ""
    words = " ".join(text.split())
    # Every run checks the catalogue and the example folders of every chapter, whatever --only says.
    assert "every chapter" in words


def test_the_default_folders_are_the_ones_of_this_worktree() -> None:
    here = Path(generate.__file__).resolve().parent
    assert here == generate.ROOT
    assert generate.ROOT == generate.REPO / "docs" / "operator_map"
    assert (generate.REPO / "src" / "vtlengine" / "__init__.py").is_file()


# --- Comparing with the committed baseline ---

ALPHA_KEYS = {"alpha/first/ds/manual_ex_1", "alpha/first/ds/two_rows", "alpha/first/sc/one_value"}


def stdout_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    """The lines that the generator printed on its standard output."""
    return capsys.readouterr().out.splitlines()


def test_compare_writes_nothing_and_passes_when_nothing_changed(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate.main([]) == 0
    before = world.files()
    capsys.readouterr()
    assert generate.main(["--compare"]) == 0
    assert world.files() == before
    lines = stdout_lines(capsys)
    assert lines[-1] == "0 regressions, 0 improvements, 0 other changes, 0 new, 0 removed"


def test_compare_fails_on_a_regression_and_names_it(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate.main([]) == 0
    before = world.files()
    world.modes["alpha"] = "wrong"
    capsys.readouterr()
    assert generate.main(["--compare"]) == 1
    lines = stdout_lines(capsys)
    assert "regression: alpha/first/ds/two_rows: 🟢 → 🟡 (pandas ❌, DuckDB ❌)" in lines
    assert lines[-1] == "3 regressions, 0 improvements, 0 other changes, 0 new, 0 removed"
    assert world.files() == before


def test_compare_reruns_the_regressions_once(world: World) -> None:
    assert generate.main([]) == 0
    world.modes["alpha"] = "wrong"
    world.calls.clear()
    assert generate.main(["--compare"]) == 1
    assert len(world.calls) == 2
    rerun = {path.relative_to(world.root / "examples").as_posix() for path, _, _ in world.calls[1]}
    assert rerun == ALPHA_KEYS


def test_compare_ignores_a_regression_that_does_not_repeat(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate.main([]) == 0
    first_run = world.run_all

    def wrong_once(
        jobs: Sequence[tuple[Path, Engine, int]], workers: int, timeout_s: float
    ) -> dict[tuple[Path, Engine], RunOutput]:
        outputs = first_run(jobs, workers, timeout_s)
        world.modes.clear()  # the second run is right again
        return outputs

    world.modes["alpha"] = "wrong"
    monkeypatch.setattr(generate, "run_all", wrong_once)
    capsys.readouterr()
    assert generate.main(["--compare"]) == 0
    lines = stdout_lines(capsys)
    assert lines[-2] == "0 regressions, 0 improvements, 0 other changes, 0 new, 0 removed"
    assert lines[-1] == (
        "3 regressions did not repeat on a second run: alpha/first/ds/manual_ex_1, "
        "alpha/first/ds/two_rows, alpha/first/sc/one_value"
    )


def test_compare_reports_improvements_and_passes(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    world.modes["beta"] = "wrong"
    assert generate.main([]) == 0
    world.modes.clear()
    capsys.readouterr()
    assert generate.main(["--compare"]) == 0
    lines = stdout_lines(capsys)
    assert "improvement: beta/second/ds/two_rows: 🟡 → 🟢 (pandas ✅, DuckDB ✅)" in lines
    assert lines[-1] == "0 regressions, 1 improvement, 0 other changes, 0 new, 0 removed"


def test_compare_does_not_count_green_to_shield(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate.main([]) == 0
    world.modes["beta"] = "refused"
    capsys.readouterr()
    assert generate.main(["--compare"]) == 0
    lines = stdout_lines(capsys)
    assert "other: beta/second/ds/two_rows: 🟢 → 🛡️ (pandas 🛡 2-1-15-6, DuckDB 🛡 2-1-15-6)" in lines
    assert lines[-1] == "0 regressions, 0 improvements, 1 other change, 0 new, 0 removed"


def test_compare_without_a_baseline_is_a_problem(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate.main(["--compare"]) == 1
    assert problems(capsys) == ["status.json: no baseline to compare with (run generate.py first)"]
    assert world.preflights == []
    assert world.calls == []


def test_compare_with_an_unreadable_baseline_is_a_problem(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    (world.root / "status.json").write_text("{", encoding="utf-8")
    assert generate.main(["--compare"]) == 1
    [line] = problems(capsys)
    assert line.startswith("status.json: cannot read the previous results (JSONDecodeError")
    assert world.calls == []


def test_compare_with_only_compares_the_named_chapters(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate.main([]) == 0
    world.modes["beta"] = "wrong"
    world.calls.clear()
    capsys.readouterr()
    assert generate.main(["--compare", "--only", "alpha"]) == 0
    assert world.run_chapters() == {"alpha"}
    lines = stdout_lines(capsys)
    assert lines[-1] == "0 regressions, 0 improvements, 0 other changes, 0 new, 0 removed"


def test_summary_appends_markdown(world: World, tmp_path: Path) -> None:
    assert generate.main([]) == 0
    summary = tmp_path / "summary.md"
    summary.write_text("before\n", encoding="utf-8")
    assert generate.main(["--compare", "--summary", str(summary)]) == 0
    assert generate.main(["--compare", "--summary", str(summary)]) == 0
    text = summary.read_text(encoding="utf-8")
    assert text.startswith("before\n")
    assert text.count("### Operator map") == 2
    assert "No example changed status." in text


def test_a_full_run_with_summary_compares_with_the_file_it_replaces(
    world: World, tmp_path: Path
) -> None:
    assert generate.main([]) == 0
    world.modes["beta"] = "wrong"
    summary = tmp_path / "summary.md"
    assert generate.main(["--summary", str(summary)]) == 0
    assert world.status()["examples"]["beta/second/ds/two_rows"]["status"] == "partial"
    text = summary.read_text(encoding="utf-8")
    assert "#### Regressions (1)" in text
    assert "beta/second/ds/two_rows" in text


def test_compare_and_check_only_cannot_go_together(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        generate.main(["--compare", "--check-only"])
    assert stopped.value.code == 2
    assert "argument --check-only: not allowed with argument --compare" in capsys.readouterr().err
