# ruff: noqa: S101
"""Checks for the copy of the examples of the map into the built site.

``docs/scripts/operator_map_site.py`` is what ``docs/conf.py`` calls when a build has finished.
It is a script of the docs, not a module of the tool, so it is loaded from its file.

The folder of an example gets a page of its own when ``operator_map/example_pages.json`` has an
entry for it, and a listing otherwise. The notes are rendered with docutils, which the main venv
does not have: the checks that need it are skipped there, and run in the venv of the docs::

    docsvenv/bin/python -m pytest -p no:cacheprovider -o python_files='check_*.py' \\
        -o pythonpath="src docs/operator_map" docs/operator_map/tool/tests/check_site.py
"""

from __future__ import annotations

import ast
import html
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

from tool.render import _MEANING, LEGEND_HTML
from tool.status import EMOJI

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "operator_map_site.py"
CONF = SCRIPT.parents[1] / "conf.py"
PAGES = "example_pages.json"
KEY = "string/length/comp/minimal"  # the first example of ``make_examples``


def load_site() -> Any:
    spec = importlib.util.spec_from_file_location("operator_map_site", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SITE = load_site()


def make_examples(srcdir: Path) -> Path:
    """A source tree with two examples; returns the ``operator_map/examples`` folder."""
    examples = srcdir / "operator_map" / "examples"
    one = examples / "string" / "length" / "comp" / "minimal"
    (one / "input").mkdir(parents=True)
    (one / "expected").mkdir()
    (one / "script.vtl").write_text("DS_r := length(DS_1);\n", encoding="utf-8")
    (one / "input" / "DS_1.csv").write_text("Id_1,Me_1\n1,a\n", encoding="utf-8")
    (one / "expected" / "DS_r.csv").write_text("Id_1,Me_1\n1,1\n", encoding="utf-8")
    (one / "notes.md").write_text("Notes.\n", encoding="utf-8")
    two = examples / "general" / "non_persistent_assignment" / "sc" / "minimal"
    two.mkdir(parents=True)
    (two / "script.vtl").write_text("sc_r := 1;\n", encoding="utf-8")
    return examples


def links(index: Path) -> list[str]:
    """The targets of the links of a listing, in order."""
    return re.findall(r'<a href="([^"]*)"', index.read_text(encoding="utf-8"))


def shown(index: Path) -> list[tuple[str, str]]:
    """The files that a page shows, in order: the heading and the text of each, unescaped."""
    pairs = re.findall(r"<h2>([^<]*)</h2>\n<pre>(.*?)</pre>", index.read_text("utf-8"), re.DOTALL)
    return [(html.unescape(name), html.unescape(text)) for name, text in pairs]


def page_entry(**changes: str) -> dict[str, str]:
    """The metadata of an example as ``generate.py`` writes it, with ``changes`` on top."""
    entry = {
        "title": "Minimal",
        "status": "ok",
        "pandas": "✅",
        "duckdb": "✅",
        "operator_page": "string/length.html",
        "anchor": "ex-comp-minimal",
    }
    return {**entry, **changes}


def write_pages(src: Path, pages: Any) -> None:
    """Write ``operator_map/example_pages.json`` of ``src``."""
    document = json.dumps(pages, ensure_ascii=False)
    (src / "operator_map" / PAGES).write_text(document, encoding="utf-8")


def build(src: Path, out: Path, pages: Any = None) -> Path:
    """Copy the examples of ``src`` to ``out``, with the metadata ``pages`` if there are any.

    Returns the folder of the example ``KEY`` in ``out``.
    """
    if pages is not None:
        write_pages(src, pages)
    assert SITE.copy_examples(src, out) is True
    return out / "operator_map" / "examples" / KEY


def hrefs(page: Path) -> list[str]:
    """The targets of the links of a page, in order, whatever the other attributes of a link."""
    return re.findall(r'<a\b[^>]*?\bhref="([^"]*)"', page.read_text(encoding="utf-8"))


def sections(page: Path) -> dict[str, str]:
    """What a page of an example shows of each file: the path in its heading, then its HTML."""
    found = re.findall(
        r'<h2>(.*?) <a class="dl"[^>]*>.*?</a></h2>\n(.*?)(?=<h2>|</body>)',
        page.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    return {html.unescape(name): body for name, body in found}


def pre(section: str) -> str:
    """The text in the ``<pre>`` of a section, unescaped."""
    found = re.search(r"<pre>(.*?)</pre>", section, re.DOTALL)
    assert found is not None, section
    return html.unescape(found.group(1))


def cells(section: str) -> list[list[str]]:
    """The rows of the table of a section, the header first, each cell as text."""
    rows = re.findall(r"<tr>(.*?)</tr>", section, re.DOTALL)
    return [
        [html.unescape(cell) for cell in re.findall(r"<t[dh]>(.*?)</t[dh]>", row, re.DOTALL)]
        for row in rows
    ]


def test_the_examples_are_copied_next_to_the_pages(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    make_examples(src)
    assert SITE.copy_examples(src, out) is True
    copied = out / "operator_map" / "examples" / "string" / "length" / "comp" / "minimal"
    assert (copied / "script.vtl").read_text(encoding="utf-8") == "DS_r := length(DS_1);\n"
    assert (copied / "input" / "DS_1.csv").is_file()
    assert (copied / "expected" / "DS_r.csv").is_file()
    assert (copied / "notes.md").is_file()
    assert (out / "operator_map" / "examples" / "general" / "non_persistent_assignment").is_dir()


def test_a_source_without_examples_copies_nothing(tmp_path: Path) -> None:
    # An older version of the docs has no operator map at all.
    (tmp_path / "src" / "operator_map").mkdir(parents=True)
    assert SITE.copy_examples(tmp_path / "src", tmp_path / "out") is False
    assert SITE.copy_examples(tmp_path / "nothing", tmp_path / "out") is False
    assert not (tmp_path / "out").exists()


def test_every_folder_gets_a_listing_that_links_what_it_holds(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    make_examples(src)
    SITE.copy_examples(src, out)
    root = out / "operator_map" / "examples"
    assert links(root / "index.html") == ["general/", "string/"]
    example = root / "string" / "length" / "comp" / "minimal"
    # A way up, the folders, and then the files.
    assert links(example / "index.html") == [
        "../",
        "expected/",
        "input/",
        "notes.md",
        "script.vtl",
    ]
    assert links(example / "input" / "index.html") == ["../", "DS_1.csv"]
    assert links(example / "expected" / "index.html") == ["../", "DS_r.csv"]
    # Every folder has one, whatever its depth.
    folders = [path for path in root.rglob("*") if path.is_dir()]
    assert len(folders) == 10
    assert all((folder / "index.html").is_file() for folder in [root, *folders])
    listing = (example / "index.html").read_text(encoding="utf-8")
    assert "<title>examples/string/length/comp/minimal</title>" in listing
    assert listing.startswith("<!DOCTYPE html>")


def test_the_page_of_an_example_shows_every_file_of_the_example(tmp_path: Path) -> None:
    # A static host may serve a .vtl or a .csv file as a download: its page shows the text.
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    (one / "value_domains").mkdir()
    (one / "value_domains" / "VD_1.json").write_text('{"name": "VD_1"}\n', encoding="utf-8")
    (one / "external_routines").mkdir()
    query = '{"query": "SELECT * FROM DS_1 WHERE Me_1 < 2 AND Me_2 = \'a&b\'"}\n'
    (one / "external_routines" / "SQL_1.json").write_text(query, encoding="utf-8")
    SITE.copy_examples(src, out)
    example = out / "operator_map" / "examples" / "string" / "length" / "comp" / "minimal"
    # The script, the inputs, the expected outputs, the other inputs of ``run``, the notes.
    assert shown(example / "index.html") == [
        ("script.vtl", "DS_r := length(DS_1);\n"),
        ("input/DS_1.csv", "Id_1,Me_1\n1,a\n"),
        ("expected/DS_r.csv", "Id_1,Me_1\n1,1\n"),
        ("value_domains/VD_1.json", '{"name": "VD_1"}\n'),
        ("external_routines/SQL_1.json", query),
        ("notes.md", "Notes.\n"),
    ]
    page = (example / "index.html").read_text(encoding="utf-8")
    assert "Me_1 &lt; 2 AND Me_2 = &#x27;a&amp;b&#x27;" in page  # as text, never as markup
    # The links to the files are still there, and a folder of the example shows its own files.
    assert links(example / "index.html") == [
        "../",
        "expected/",
        "external_routines/",
        "input/",
        "value_domains/",
        "notes.md",
        "script.vtl",
    ]
    assert shown(example / "input" / "index.html") == [("DS_1.csv", "Id_1,Me_1\n1,a\n")]
    assert shown(example / "expected" / "index.html") == [("DS_r.csv", "Id_1,Me_1\n1,1\n")]


def test_the_folders_above_the_examples_show_no_file(tmp_path: Path) -> None:
    # They hold examples, not files: showing them all would put every example in one page.
    src, out = tmp_path / "src", tmp_path / "out"
    make_examples(src)
    SITE.copy_examples(src, out)
    root = out / "operator_map" / "examples"
    for folder in (root, root / "string", root / "string" / "length" / "comp"):
        assert "<pre>" not in (folder / "index.html").read_text(encoding="utf-8"), folder
    two = root / "general" / "non_persistent_assignment" / "sc" / "minimal"
    assert shown(two / "index.html") == [("script.vtl", "sc_r := 1;\n")]


def test_a_file_too_large_or_not_text_is_linked_instead_of_shown(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    (one / "input" / "DS_2.csv").write_text("x" * (64 * 1024 + 1), encoding="utf-8")
    (one / "input" / "DS_3.csv").write_bytes(b"\xff\xfe not utf-8")
    SITE.copy_examples(src, out)
    page = (
        out / "operator_map" / "examples" / "string" / "length" / "comp" / "minimal" / "index.html"
    )
    text = page.read_text(encoding="utf-8")
    assert [name for name, _ in shown(page)] == [
        "script.vtl",
        "input/DS_1.csv",
        "expected/DS_r.csv",
        "notes.md",
    ]
    assert (
        "<h2>input/DS_2.csv</h2>\n<p>Not shown here: 65537 bytes, more than 65536. "
        '<a href="input/DS_2.csv">Open the file</a>.</p>'
    ) in text
    assert (
        "<h2>input/DS_3.csv</h2>\n<p>Not shown here: it is not UTF-8 text. "
        '<a href="input/DS_3.csv">Open the file</a>.</p>'
    ) in text


def test_the_diagrams_run_at_the_antiscript_security_level() -> None:
    # Their clicks are links that the site checks; scripts in a diagram never run.
    tree = ast.parse(CONF.read_text(encoding="utf-8"))
    [config] = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "mermaid_init_config" for t in node.targets)
    ]
    assert config["securityLevel"] == "antiscript"


def test_an_index_that_the_source_has_is_kept(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    examples = make_examples(src)
    (examples / "string" / "index.html").write_text("<p>mine</p>", encoding="utf-8")
    SITE.copy_examples(src, out)
    copied = out / "operator_map" / "examples" / "string" / "index.html"
    assert copied.read_text(encoding="utf-8") == "<p>mine</p>"


def test_names_are_escaped_in_the_listing(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    folder = make_examples(src) / "string"
    (folder / "a b&c.txt").write_text("x", encoding="utf-8")
    (folder / "<i>.txt").write_text("x", encoding="utf-8")
    SITE.copy_examples(src, out)
    text = (out / "operator_map" / "examples" / "string" / "index.html").read_text(encoding="utf-8")
    assert '<a href="a%20b%26c.txt">a b&amp;c.txt</a>' in text
    assert '<a href="%3Ci%3E.txt">&lt;i&gt;.txt</a>' in text
    assert "<i>" not in text


def test_a_second_build_into_the_same_output_updates_the_listings(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    examples = make_examples(src)
    SITE.copy_examples(src, out)
    (examples / "string" / "extra.txt").write_text("x", encoding="utf-8")
    assert SITE.copy_examples(src, out) is True
    index = out / "operator_map" / "examples" / "string" / "index.html"
    assert links(index) == ["../", "length/", "extra.txt"]


# --- The page of an example folder ---


def test_example_page_renders_notes_fields(tmp_path: Path) -> None:
    pytest.importorskip("docutils.core")
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    (one / "notes.md").unlink()
    (one / "notes.rst").write_text(
        ":Tests: ``length`` of a *string*, not <b>markup</b> & more.\n"
        ':Manual: `Length <https://example.org/length>`__: "the number of characters".\n',
        encoding="utf-8",
    )
    page = build(src, out, {KEY: page_entry()}) / "index.html"
    text = page.read_text(encoding="utf-8")
    # The fields are rendered as a field list: "Tests" is a name, not the text ":Tests:".
    assert re.search(r"<dt[^>]*>Tests\b", text)
    assert re.search(r"<dt[^>]*>Manual\b", text)
    assert ":Tests:" not in text
    assert ":Manual:" not in text
    # The markup of RST is read, and what looks like HTML in a note stays text.
    assert "<em>string</em>" in text
    assert "``" not in text
    assert 'href="https://example.org/length"' in text
    assert "not &lt;b&gt;markup&lt;/b&gt; &amp; more." in text
    assert "<b>" not in text
    # The notes are not one of the files of the page.
    assert list(sections(page)) == ["script.vtl", "input/DS_1.csv", "expected/DS_r.csv"]


def test_example_page_csv_table_escaping(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    (one / "input" / "DS_1.csv").write_text(
        'Id_1,Me_1,Me_2\n1,"a,b","x""y"\n2,,<i>&amp;</i>\n3," padded ",\'q\'\n4,"two\nlines",\n',
        encoding="utf-8",
    )
    (one / "expected" / "DS_r.csv").write_text("Id_1,Me_1\n", encoding="utf-8")  # a header only
    page = build(src, out, {KEY: page_entry()}) / "index.html"
    found = sections(page)
    # A cell is what the file says: a comma and a quote in a quoted cell, an empty cell, the
    # spaces around a value, a line break.
    assert cells(found["input/DS_1.csv"]) == [
        ["Id_1", "Me_1", "Me_2"],
        ["1", "a,b", 'x"y'],
        ["2", "", "<i>&amp;</i>"],
        ["3", " padded ", "'q'"],
        ["4", "two\nlines", ""],
    ]
    # It is text, never markup, and a table, not a block of text.
    assert "<td>&lt;i&gt;&amp;amp;&lt;/i&gt;</td>" in found["input/DS_1.csv"]
    assert "<i>" not in page.read_text(encoding="utf-8")
    assert "<pre>" not in found["input/DS_1.csv"]
    # A file with a header only is a table with no row, and says so.
    assert cells(found["expected/DS_r.csv"]) == [["Id_1", "Me_1"]]
    assert "no data points" in found["expected/DS_r.csv"]


def test_example_page_csv_table_drops_a_byte_order_mark(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    (one / "input" / "DS_1.csv").write_bytes(b"\xef\xbb\xbfId_1,Me_1\r\n1,a\r\n")
    page = build(src, out, {KEY: page_entry()}) / "index.html"
    # The mark is not a character of the name of the first column.
    assert cells(sections(page)["input/DS_1.csv"]) == [["Id_1", "Me_1"], ["1", "a"]]


@pytest.mark.parametrize("notes", ["notes.rst", "notes.md"])
def test_example_page_no_notes_link(tmp_path: Path, notes: str) -> None:
    if notes == "notes.rst":
        pytest.importorskip("docutils.core")
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    (one / "notes.md").unlink()
    (one / notes).write_text(":Tests: the length.\n", encoding="utf-8")
    page = build(src, out, {KEY: page_entry()}) / "index.html"
    text = page.read_text(encoding="utf-8")
    # The notes are on the page: neither linked nor listed as a file.
    assert "notes.rst" not in text
    assert "notes.md" not in text
    assert "the length." in text
    assert list(sections(page)) == ["script.vtl", "input/DS_1.csv", "expected/DS_r.csv"]
    # The way back, and a link to download each of the files.
    assert hrefs(page) == [
        "../../../../../string/length.html#ex-comp-minimal",
        "script.vtl",
        "input/DS_1.csv",
        "expected/DS_r.csv",
    ]


def test_example_page_back_link(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    make_examples(src)
    target = out / "operator_map" / "string" / "length.html"
    target.parent.mkdir(parents=True)
    target.write_text('<details id="ex-comp-minimal"></details>', encoding="utf-8")
    stylesheet = out / "_static" / "operator_map.css"
    stylesheet.parent.mkdir()
    stylesheet.write_text(".om-chip {}", encoding="utf-8")
    page = build(src, out, {KEY: page_entry()}) / "index.html"
    text = page.read_text(encoding="utf-8")
    # The page is five folders below the folder of the map, and six below the root of the site.
    back = "../../../../../string/length.html#ex-comp-minimal"
    assert f'<a href="{back}">← length operator page</a>' in text
    path, _, fragment = back.partition("#")
    assert (page.parent / path).resolve() == target.resolve()
    assert f'id="{fragment}"' in target.read_text(encoding="utf-8")
    [css] = re.findall(r'<link rel="stylesheet" href="([^"]*)">', text)
    assert css == "../../../../../../_static/operator_map.css"
    assert (page.parent / css).resolve() == stylesheet.resolve()


def test_the_way_up_follows_the_depth_of_the_folder(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    make_examples(src)
    shallow = src / "operator_map" / "examples" / "odd" / "one" / "ex"
    shallow.mkdir(parents=True)
    (shallow / "script.vtl").write_text("x;\n", encoding="utf-8")
    build(src, out, {"odd/one/ex": page_entry(operator_page="odd/one.html", anchor="a")})
    text = (out / "operator_map" / "examples" / "odd" / "one" / "ex" / "index.html").read_text(
        encoding="utf-8"
    )
    assert '<a href="../../../../odd/one.html#a">' in text
    assert 'href="../../../../../_static/operator_map.css"' in text


@pytest.mark.parametrize("pages", [None, {"other/key/x/y": page_entry()}], ids=["no file", "stale"])
def test_folder_without_metadata_gets_listing(tmp_path: Path, pages: Any) -> None:
    # An older version of the docs has no example_pages.json, and a stale copy has no key for
    # the example: either way the folder keeps its listing.
    src, out = tmp_path / "src", tmp_path / "out"
    make_examples(src)
    example = build(src, out, pages)
    text = (example / "index.html").read_text(encoding="utf-8")
    assert "<h1>examples/string/length/comp/minimal/</h1>" in text
    assert links(example / "index.html") == [
        "../",
        "expected/",
        "input/",
        "notes.md",
        "script.vtl",
    ]
    assert shown(example / "index.html")[0] == ("script.vtl", "DS_r := length(DS_1);\n")


def test_only_the_folders_that_have_metadata_get_a_page(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    make_examples(src)
    one = build(src, out, {KEY: page_entry()})
    root = out / "operator_map" / "examples"
    two = root / "general" / "non_persistent_assignment" / "sc" / "minimal"
    assert "<h1>Minimal " in (one / "index.html").read_text(encoding="utf-8")
    # The other example has no metadata: its listing, with the text of its files.
    assert "<h1>examples/general/non_persistent_assignment/sc/minimal/</h1>" in (
        two / "index.html"
    ).read_text(encoding="utf-8")
    assert shown(two / "index.html") == [("script.vtl", "sc_r := 1;\n")]
    # The folders of an example, and the folders above the examples, are no example.
    assert links(one / "input" / "index.html") == ["../", "DS_1.csv"]
    assert shown(one / "input" / "index.html") == [("DS_1.csv", "Id_1,Me_1\n1,a\n")]
    assert shown(one / "expected" / "index.html") == [("DS_r.csv", "Id_1,Me_1\n1,1\n")]
    for folder in (root, root / "string", root / "string" / "length" / "comp"):
        listing = (folder / "index.html").read_text(encoding="utf-8")
        assert "<h1>examples" in listing, folder
        assert "<pre>" not in listing, folder


def test_the_statuses_are_those_of_the_operator_pages() -> None:
    # The script of the site does not import the tool: its table is kept in step by this check.
    assert {status: emoji for status, (emoji, _) in SITE.STATUSES.items()} == EMOJI
    assert {status: meaning for status, (_, meaning) in SITE.STATUSES.items()} == _MEANING


@pytest.mark.parametrize("status", list(EMOJI))
def test_example_page_chip(tmp_path: Path, status: str) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    make_examples(src)
    page = build(src, out, {KEY: page_entry(status=status)}) / "index.html"
    # The chip of the legend of the operator pages, drawn by the CSS of the map.
    chip = f'<span class="om-chip {status}">{EMOJI[status]} {_MEANING[status]}</span>'
    assert chip in LEGEND_HTML
    assert f"<h1>Minimal {chip}</h1>" in page.read_text(encoding="utf-8")


def test_example_page_marks(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    make_examples(src)
    pages = {KEY: page_entry(pandas="🛡 2-1-15-6", duckdb="💥 AttributeError")}
    page = build(src, out, pages) / "index.html"
    marks = (
        '<p class="marks">pandas 🛡 <code>2-1-15-6</code> · '
        "DuckDB 💥 <code>AttributeError</code></p>"
    )
    assert marks in page.read_text(encoding="utf-8")
    page = build(src, out, {KEY: page_entry(pandas="✅", duckdb="❌")}) / "index.html"
    assert '<p class="marks">pandas ✅ · DuckDB ❌</p>' in page.read_text(encoding="utf-8")


def test_example_page_shows_notes_md_as_text(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    notes = "# `length`\n\n*Tests* <b>this</b> & that.\n\n| a | b |\n"
    (one / "notes.md").write_text(notes, encoding="utf-8")
    page = build(src, out, {KEY: page_entry()}) / "index.html"
    text = page.read_text(encoding="utf-8")
    # Preformatted text, escaped: the markdown is not read, and nothing in it is markup.
    assert '<div class="notes"><pre>' in text
    assert pre(text.split('<div class="notes">', 1)[1]) == notes
    assert "<b>" not in text
    assert "notes.md" not in text


def test_example_page_without_notes_has_no_notes_section(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    (one / "notes.md").unlink()
    page = build(src, out, {KEY: page_entry()}) / "index.html"
    assert "notes" not in page.read_text(encoding="utf-8").split("<body>", 1)[1]
    assert list(sections(page)) == ["script.vtl", "input/DS_1.csv", "expected/DS_r.csv"]


def test_notes_rst_wins_over_notes_md(tmp_path: Path) -> None:
    pytest.importorskip("docutils.core")
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    (one / "notes.rst").write_text(":Tests: the rst.\n", encoding="utf-8")
    (one / "notes.md").write_text("the markdown\n", encoding="utf-8")
    text = (build(src, out, {KEY: page_entry()}) / "index.html").read_text(encoding="utf-8")
    assert "the rst." in text
    assert "the markdown" not in text


def test_example_page_shows_every_file_with_a_download_link(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    files = {
        "input/DS_1.json": '{"name": "DS_1", "type": "String"}\n',
        "expected/scalars.json": '{"scalars": [{"name": "SC_r", "value": "a&b"}]}\n',
        "value_domains/VD_1.json": '{"name": "VD_1"}\n',
        "external_routines/SQL_1.json": '{"query": "SELECT * FROM DS_1 WHERE Me_1 < 2"}\n',
        "input/.hidden": "not part of the example\n",
    }
    for name, text in files.items():
        (one / name).parent.mkdir(exist_ok=True)
        (one / name).write_text(text, encoding="utf-8")
    page = build(src, out, {KEY: page_entry()}) / "index.html"
    found = sections(page)
    # The script, the inputs, what is expected, the other inputs of ``run``; not the notes.
    assert list(found) == [
        "script.vtl",
        "input/DS_1.csv",
        "input/DS_1.json",
        "expected/DS_r.csv",
        "expected/scalars.json",
        "value_domains/VD_1.json",
        "external_routines/SQL_1.json",
    ]
    # The script and the JSON are text, exactly as written; a CSV is a table.
    assert pre(found["script.vtl"]) == "DS_r := length(DS_1);\n"
    for name, text in files.items():
        if name != "input/.hidden":
            assert pre(found[name]) == text, name
    assert "<table>" in found["input/DS_1.csv"]
    assert "<pre>" not in found["input/DS_1.csv"]
    # Each of them has a link to download it.
    text = page.read_text(encoding="utf-8")
    assert re.findall(r'<a class="dl" href="([^"]*)" download', text) == list(found)


def test_example_page_links_a_file_that_is_too_large_or_not_text(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    (one / "input" / "DS_2.csv").write_text("x" * (64 * 1024 + 1), encoding="utf-8")
    (one / "input" / "DS_3.csv").write_bytes(b"\xff\xfe not utf-8")
    page = build(src, out, {KEY: page_entry()}) / "index.html"
    found = sections(page)
    assert list(found) == [
        "script.vtl",
        "input/DS_1.csv",
        "input/DS_2.csv",
        "input/DS_3.csv",
        "expected/DS_r.csv",
    ]
    assert "Not shown here: 65537 bytes, more than 65536." in found["input/DS_2.csv"]
    assert "Not shown here: it is not UTF-8 text." in found["input/DS_3.csv"]
    for name in ("input/DS_2.csv", "input/DS_3.csv"):
        assert "<pre>" not in found[name]
        assert "<table" not in found[name]
    # They can still be downloaded.
    assert 'href="input/DS_2.csv" download' in page.read_text(encoding="utf-8")
    assert 'href="input/DS_3.csv" download' in page.read_text(encoding="utf-8")


def test_example_page_escapes_what_comes_from_the_files_and_the_metadata(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    (one / "script.vtl").write_text(
        "DS_r := DS_1[filter Me_1 < \"<b>\" & 'x'];\n", encoding="utf-8"
    )
    (one / "input" / "a b&c.txt").write_text("<i>&</i>\n", encoding="utf-8")
    entry = page_entry(
        title='<i>"T"</i> & co',
        pandas="💥 <Boom&>",
        operator_page="a b/c&d.html",
        anchor='x"y#z',
    )
    page = build(src, out, {KEY: entry}) / "index.html"
    text = page.read_text(encoding="utf-8")
    assert "<i>" not in text
    assert "<b>" not in text
    assert "<Boom" not in text
    # The title, in the page and above it; the mark, as a code.
    assert "<title>&lt;i&gt;&quot;T&quot;&lt;/i&gt; &amp; co · string/length/comp</title>" in text
    assert "<h1>&lt;i&gt;&quot;T&quot;&lt;/i&gt; &amp; co " in text
    assert "💥 <code>&lt;Boom&amp;&gt;</code>" in text
    # The URLs are quoted, and so is the name of a file in its link; in the text it is escaped.
    assert 'href="../../../../../a%20b/c%26d.html#x%22y%23z"' in text
    assert 'href="input/a%20b%26c.txt" download' in text
    found = sections(page)
    assert pre(found["input/a b&c.txt"]) == "<i>&</i>\n"
    assert "<h2>input/a b&amp;c.txt " in text
    assert pre(found["script.vtl"]) == "DS_r := DS_1[filter Me_1 < \"<b>\" & 'x'];\n"


def test_an_example_index_that_the_source_has_is_kept(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    (one / "index.html").write_text("<p>mine</p>", encoding="utf-8")
    page = build(src, out, {KEY: page_entry()}) / "index.html"
    assert page.read_text(encoding="utf-8") == "<p>mine</p>"


def test_a_second_build_updates_the_page_of_an_example(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    one = make_examples(src) / "string" / "length" / "comp" / "minimal"
    page = build(src, out, {KEY: page_entry()}) / "index.html"
    assert "om-chip ok" in page.read_text(encoding="utf-8")
    (one / "input" / "DS_1.csv").write_text("Id_1,Me_1\n9,z\n", encoding="utf-8")
    build(src, out, {KEY: page_entry(status="diverge", title="Changed")})
    text = page.read_text(encoding="utf-8")
    assert "om-chip diverge" in text
    assert "om-chip ok" not in text
    assert cells(sections(page)["input/DS_1.csv"]) == [["Id_1", "Me_1"], ["9", "z"]]


def test_metadata_that_is_not_json_stops_the_build(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    make_examples(src)
    (src / "operator_map" / PAGES).write_text('{"string/length', encoding="utf-8")
    with pytest.raises(ValueError, match=PAGES):
        SITE.copy_examples(src, out)
    assert not out.exists()  # nothing was copied


@pytest.mark.parametrize(
    ("pages", "complaint"),
    [
        ([page_entry()], "an object"),
        ({KEY: "ok"}, KEY),
        ({KEY: {"title": "Minimal"}}, KEY),
        ({KEY: page_entry(status="great")}, "status"),
        ({KEY: {**page_entry(), "anchor": None}}, "anchor"),
    ],
    ids=["a list", "not an entry", "missing fields", "unknown status", "not text"],
)
def test_an_entry_that_is_not_a_page_stops_the_build(
    tmp_path: Path, pages: Any, complaint: str
) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    make_examples(src)
    write_pages(src, pages)
    with pytest.raises(ValueError, match=complaint):
        SITE.copy_examples(src, out)
    assert not out.exists()
