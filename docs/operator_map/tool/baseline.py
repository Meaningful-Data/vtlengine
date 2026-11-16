"""A run of the operator map against its committed baseline: what moved, and the report of it.

The baseline is the ``status.json`` of the map, the status of each example as it was committed.
A run ranks the new status of each example against it (``RANK``): a status that ranks lower is a
regression, one that ranks higher an improvement. 🟢 and 🛡️ rank alike, as a block of them only
is 🟢: a change between them is another change, and never a regression. An example that the
baseline does not have is new, one that the run does not have is removed; neither is a
regression. An example whose status stays is no change, whatever its details.

``report_lines`` says what moved, one line each, for the terminal; ``summary_markdown`` says it
as Markdown, for the summary of a job of GitHub Actions, with a link from each example to its
folder at the commit of the run.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from tool.model import Status
from tool.render import RunMeta
from tool.status import EMOJI, EngineVerdict, ExampleResult, counts, mark

RANK: Final[dict[Status, int]] = {"ok": 4, "refused": 4, "diverge": 3, "partial": 2, "missing": 1}
EXAMPLES_URL: Final = (
    "https://github.com/Meaningful-Data/vtlengine/tree/{commit}/docs/operator_map/examples/{key}"
)
_DETAIL_MAX: Final = 200  # the characters of a detail in a table, as on the pages of the map
_NONE: Final = "—"  # a cell with nothing: the status before a new example, after a removed one


@dataclass(frozen=True)
class Change:
    """An example whose status moved: before is ``None`` for a new one, after for a removed one."""

    key: str
    before: ExampleResult | None
    after: ExampleResult | None


@dataclass(frozen=True)
class Comparison:
    """What moved between the baseline and a run, by kind, each sorted by key."""

    regressions: tuple[Change, ...]
    improvements: tuple[Change, ...]
    other: tuple[Change, ...]  # 🟢 ↔ 🛡️
    new: tuple[Change, ...]
    removed: tuple[Change, ...]


def compare(
    baseline: Mapping[str, ExampleResult], results: Mapping[str, ExampleResult]
) -> Comparison:
    """What moved from ``baseline`` to ``results``, both by example key."""
    kinds: dict[str, list[Change]] = {
        kind: [] for kind in ("regressions", "improvements", "other", "new", "removed")
    }
    for key in sorted(baseline.keys() | results.keys()):
        before, after = baseline.get(key), results.get(key)
        if before is None:
            kinds["new"].append(Change(key, None, after))
        elif after is None:
            kinds["removed"].append(Change(key, before, None))
        elif before.status != after.status:
            step = RANK[after.status] - RANK[before.status]
            kind = "regressions" if step < 0 else "improvements" if step > 0 else "other"
            kinds[kind].append(Change(key, before, after))
    return Comparison(**{kind: tuple(changes) for kind, changes in kinds.items()})


def report_lines(comparison: Comparison) -> list[str]:
    """A line for each change, then one with how many of each kind there are."""
    lines = [
        f"{label}: {change.key}: {_move(change)}{_marks_in_brackets(change.after)}"
        for label, changes in _labelled(comparison)
        for change in changes
    ]
    lines.append(
        ", ".join(
            [
                _count(len(comparison.regressions), "regression"),
                _count(len(comparison.improvements), "improvement"),
                _count(len(comparison.other), "other change"),
                f"{len(comparison.new)} new",
                f"{len(comparison.removed)} removed",
            ]
        )
    )
    return lines


def summary_markdown(
    comparison: Comparison,
    results: Mapping[str, ExampleResult],
    meta: RunMeta,
    *,
    limit: int = 100,
) -> str:
    """The report as Markdown: the run, its counts, then a table of each kind of change.

    A table has at most ``limit`` rows, so that a summary stays small whatever moved.
    """
    tally = counts([result.status for result in results.values()])
    parts = [
        "### Operator map",
        f"vtlengine {meta.vtlengine_version} ({meta.vtlengine_commit}) · pandas "
        f"{meta.pandas_version} · DuckDB {meta.duckdb_version}",
        " · ".join(f"{EMOJI[status]} {number}" for status, number in tally.items()),
    ]
    sections = [
        ("Regressions", comparison.regressions),
        ("Improvements", comparison.improvements),
        ("Other changes, 🟢 ↔ 🛡️", comparison.other),
        ("New examples", comparison.new),
        ("Removed examples", comparison.removed),
    ]
    if not any(changes for _, changes in sections):
        parts.append("No example changed status.")
    for title, changes in sections:
        if changes:
            parts.append(_section(title, changes, meta.vtlengine_commit, limit))
    if comparison.improvements:
        parts.append(
            "To record the improvements, run `docs/operator_map/generate.py` and commit the map."
        )
    return "\n\n".join(parts) + "\n"


def _labelled(comparison: Comparison) -> list[tuple[str, tuple[Change, ...]]]:
    return [
        ("regression", comparison.regressions),
        ("improvement", comparison.improvements),
        ("other", comparison.other),
        ("new", comparison.new),
        ("removed", comparison.removed),
    ]


def _move(change: Change) -> str:
    """``🟢 → 🟡``, or the one status of a new or a removed example."""
    statuses = [EMOJI[result.status] for result in (change.before, change.after) if result]
    return " → ".join(statuses)


def _marks_in_brackets(result: ExampleResult | None) -> str:
    return f" (pandas {mark(result.pandas)}, DuckDB {mark(result.duckdb)})" if result else ""


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _section(title: str, changes: Sequence[Change], commit: str, limit: int) -> str:
    """A titled table of ``changes``, cut at ``limit`` rows."""
    head = [
        f"#### {title} ({len(changes)})",
        "",
        "| Example | Before | After | pandas | DuckDB | Detail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows = [_row(change, commit) for change in changes[:limit]]
    more = [f"\n… and {len(changes) - limit} more"] if len(changes) > limit else []
    return "\n".join(head + rows + more)


def _row(change: Change, commit: str) -> str:
    link = f"[{change.key}]({EXAMPLES_URL.format(commit=commit, key=change.key)})"
    before = EMOJI[change.before.status] if change.before else _NONE
    after = change.after
    cells = [
        link,
        before,
        EMOJI[after.status] if after else _NONE,
        mark(after.pandas) if after else _NONE,
        mark(after.duckdb) if after else _NONE,
        _detail(after.pandas, after.duckdb) if after else "",
    ]
    return "| " + " | ".join(cells) + " |"


def _detail(pandas: EngineVerdict, duckdb: EngineVerdict) -> str:
    """The detail of the engines, as one cell: once when they say the same, else each one's."""
    if pandas.detail == duckdb.detail:
        text = pandas.detail
    else:
        text = "; ".join(
            f"{name}: {verdict.detail}"
            for name, verdict in (("pandas", pandas), ("DuckDB", duckdb))
            if verdict.detail
        )
    one_line = " ".join(text.split())
    if len(one_line) > _DETAIL_MAX:
        one_line = one_line[: _DETAIL_MAX - 1] + "…"
    return html.escape(one_line, quote=False).replace("|", "\\|")
