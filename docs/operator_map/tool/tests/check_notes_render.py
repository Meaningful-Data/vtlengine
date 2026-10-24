# ruff: noqa: S101
"""Checks that the notes of the examples are RST that docutils reads without a complaint.

``check_notes`` of ``tool.examples`` checks the layout of a ``notes.rst`` with a regular
expression, as the tool does not depend on docutils. These checks parse every
``examples/**/notes.rst`` with docutils, as the pages will include it: each must read without a
single system message of level 2 (a warning) or above. Run them in the venv of the docs, which
has docutils; without it they are skipped::

    docsvenv/bin/python -m pytest -p no:cacheprovider -o python_files='check_*.py' \\
        -o pythonpath="src docs/operator_map" docs/operator_map/tool/tests/check_notes_render.py

The field list of the notes is read as a field list, not as the bibliographic fields of a
document (``docinfo_xform`` off): the notes are included in a page, never a page of their own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

core = pytest.importorskip("docutils.core")

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def parse_messages(text: str, source: str = "notes.rst") -> list[str]:
    """What docutils says about ``text``, at level 2 (a warning) and above: nothing if clean."""
    messages: list[str] = []

    class Collector:
        def write(self, message: str) -> None:
            messages.append(message)

    settings: dict[str, Any] = {
        "report_level": 2,
        "halt_level": 5,
        "docinfo_xform": False,
        "warning_stream": Collector(),
    }
    core.publish_doctree(text, source_path=source, settings_overrides=settings)
    return [line for message in messages for line in message.splitlines() if line.strip()]


def test_all_notes_parse() -> None:
    problems = {
        path.relative_to(EXAMPLES).as_posix(): messages
        for path in sorted(EXAMPLES.rglob("notes.rst"))
        if (messages := parse_messages(path.read_text(encoding="utf-8-sig"), str(path)))
    }
    assert problems == {}


def test_notes_with_the_characters_of_rst_parse_clean() -> None:
    text = (
        ":Tests: \"quotes\", 'single' and *stars*, ``code`` and `cited`, a | pipe and a "
        "lone * star.\n"
        ":Manual: The rule :: in the middle of a sentence, and a_name_with_underscores.\n"
        ':Expected: ``DS_1#Me_1 || "x"`` gives ``2.5``; 1-1-1-1 -- a dash... and a\n'
        "   line that goes on.\n"
    )
    assert parse_messages(text) == []


@pytest.mark.parametrize(
    ("text", "complaint"),
    [
        (":Tests: *unclosed emphasis.\n", "Inline emphasis start-string without end-string"),
        (":Tests: ``unclosed literal.\n", "Inline literal start-string without end-string"),
        (":Tests: A rule::\n:Manual: M.\n", "Literal block expected; none found"),
        (":Tests: |undefined| substitution.\n", 'Undefined substitution referenced: "undefined"'),
        (":Tests: A field\nthat is not indented.\n", "Field list ends without a blank line"),
    ],
)
def test_malformed_notes_are_reported(text: str, complaint: str) -> None:
    messages = parse_messages(text)
    assert any(complaint in message for message in messages), messages
