# ruff: noqa: S101
"""Checks for the copy of the examples of the map into the built site.

``docs/scripts/operator_map_site.py`` is what ``docs/conf.py`` calls when a build has finished.
It is a script of the docs, not a module of the tool, so it is loaded from its file.
"""

from __future__ import annotations

import ast
import html
import importlib.util
import re
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "operator_map_site.py"
CONF = SCRIPT.parents[1] / "conf.py"


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
