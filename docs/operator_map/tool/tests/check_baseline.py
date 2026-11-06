# ruff: noqa: S101
"""Checks of tool.baseline: a run of the map against its committed baseline, and the report."""

from __future__ import annotations

import pytest

from tool.baseline import RANK, Change, Comparison, compare, report_lines, summary_markdown
from tool.model import Status
from tool.render import RunMeta
from tool.status import EngineVerdict, ExampleResult

MATCH = EngineVerdict("match", "")
WRONG = EngineVerdict("mismatch", "row Id_1=1: Me_1 expected 'a' got 'b'")
GUARDED = EngineVerdict("error", "RunTimeError 2-1-15-6: zero", "RunTimeError", "2-1-15-6")
CRASH = EngineVerdict("error", "KeyError: 'Me_1'", "KeyError")
META = RunMeta(
    date="2026-11-03",
    vtlengine_commit="abc1234",
    vtlengine_version="9.9.9",
    spec_commit="1bea1153",
    pandas_version="2.3.3",
    duckdb_version="1.5.5",
)
LINK = "https://github.com/Meaningful-Data/vtlengine/tree/abc1234/docs/operator_map/examples/"


def result(status: Status, pandas: EngineVerdict | None = None) -> ExampleResult:
    """A result of ``status``, with the verdicts that give it; ``pandas`` replaces the first."""
    verdicts = {
        "ok": (MATCH, MATCH),
        "refused": (GUARDED, GUARDED),
        "partial": (WRONG, WRONG),
        "missing": (CRASH, CRASH),
        "diverge": (MATCH, CRASH),
    }[status]
    first, second = verdicts
    return ExampleResult(pandas or first, second, status != "diverge", status)


def statuses(changes: tuple[Change, ...]) -> list[tuple[str, str | None, str | None]]:
    """The key, and the status before and after, of each change."""
    return [
        (
            change.key,
            change.before.status if change.before else None,
            change.after.status if change.after else None,
        )
        for change in changes
    ]


def test_rank_orders_the_statuses() -> None:
    assert RANK == {"ok": 4, "refused": 4, "diverge": 3, "partial": 2, "missing": 1}


def test_each_rank_change_is_a_regression_or_an_improvement() -> None:
    baseline = {
        "a/ok_partial": result("ok"),
        "b/refused_diverge": result("refused"),
        "c/partial_missing": result("partial"),
        "d/missing_partial": result("missing"),
        "e/diverge_ok": result("diverge"),
    }
    run = {
        "a/ok_partial": result("partial"),
        "b/refused_diverge": result("diverge"),
        "c/partial_missing": result("missing"),
        "d/missing_partial": result("partial"),
        "e/diverge_ok": result("ok"),
    }
    found = compare(baseline, run)
    assert statuses(found.regressions) == [
        ("a/ok_partial", "ok", "partial"),
        ("b/refused_diverge", "refused", "diverge"),
        ("c/partial_missing", "partial", "missing"),
    ]
    assert statuses(found.improvements) == [
        ("d/missing_partial", "missing", "partial"),
        ("e/diverge_ok", "diverge", "ok"),
    ]
    assert found.other == found.new == found.removed == ()


def test_green_and_shield_are_other_changes() -> None:
    baseline = {"a/x": result("ok"), "b/y": result("refused")}
    found = compare(baseline, {"a/x": result("refused"), "b/y": result("ok")})
    assert statuses(found.other) == [("a/x", "ok", "refused"), ("b/y", "refused", "ok")]
    assert found.regressions == found.improvements == ()


def test_new_and_removed_examples_are_reported() -> None:
    found = compare({"a/old": result("ok")}, {"b/new": result("partial")})
    assert statuses(found.new) == [("b/new", None, "partial")]
    assert statuses(found.removed) == [("a/old", "ok", None)]
    assert found.regressions == found.improvements == found.other == ()


def test_same_status_with_other_details_is_no_change() -> None:
    other = EngineVerdict("mismatch", "row Id_1=2: Me_1 expected 'c' got 'd'")
    found = compare({"a/x": result("partial")}, {"a/x": result("partial", pandas=other)})
    assert found == Comparison((), (), (), (), ())


def test_report_lines_name_the_change_and_the_marks() -> None:
    found = compare(
        {"a/x": result("ok"), "b/y": result("missing"), "c/z": result("ok")},
        {"a/x": result("partial"), "b/y": result("diverge"), "d/w": result("ok")},
    )
    assert report_lines(found) == [
        "regression: a/x: 🟢 → 🟡 (pandas ❌, DuckDB ❌)",
        "improvement: b/y: 🔴 → 🔵 (pandas ✅, DuckDB 💥 KeyError)",
        "new: d/w: 🟢 (pandas ✅, DuckDB ✅)",
        "removed: c/z: 🟢",
        "1 regression, 1 improvement, 0 other changes, 1 new, 1 removed",
    ]


def test_summary_links_each_example_to_its_folder_at_the_commit() -> None:
    run = {"alpha/first/ds/two_rows": result("partial")}
    found = compare({"alpha/first/ds/two_rows": result("ok")}, run)
    text = summary_markdown(found, run, META)
    assert f"[alpha/first/ds/two_rows]({LINK}alpha/first/ds/two_rows)" in text
    assert "vtlengine 9.9.9 (abc1234) · pandas 2.3.3 · DuckDB 1.5.5" in text


def test_summary_escapes_the_details() -> None:
    odd = EngineVerdict("mismatch", "a | b\nc `d`")
    run = {"a/x": result("partial", pandas=odd)}
    text = summary_markdown(compare({"a/x": result("ok")}, run), run, META)
    rows = [line for line in text.splitlines() if line.startswith("| [a/x]")]
    assert len(rows) == 1
    assert "a \\| b c `d`" in rows[0]
    assert rows[0].count(" | ") == 5  # six cells: no pipe of the detail splits one


def test_summary_stops_each_table_at_the_limit() -> None:
    keys = [f"a/x{n:03}" for n in range(150)]
    run = {key: result("partial") for key in keys}
    found = compare({key: result("ok") for key in keys}, run)
    text = summary_markdown(found, run, META, limit=100)
    assert sum(1 for line in text.splitlines() if line.startswith("| [a/x")) == 100
    assert "… and 50 more" in text


def test_summary_counts_the_statuses_of_the_run() -> None:
    run = {"a/x": result("ok"), "a/y": result("ok"), "b/z": result("diverge")}
    text = summary_markdown(compare(run, run), run, META)
    assert "🟢 2 · 🟡 0 · 🛡️ 0 · 🔴 0 · 🔵 1" in text
    assert "No example changed status." in text


@pytest.mark.parametrize("before", ["ok", "refused", "diverge", "partial", "missing"])
def test_no_change_without_a_status_change(before: Status) -> None:
    assert compare({"a/x": result(before)}, {"a/x": result(before)}) == Comparison(
        (), (), (), (), ()
    )
