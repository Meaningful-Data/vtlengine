"""Serve the example files of the operator map next to its pages in the built HTML site.

The pages of ``docs/operator_map`` link, relative to themselves, to ``examples/...``: the scripts,
the input and expected data, the notes, and the folders that hold them. Those files are not
sources of Sphinx, so the build copies the folder to its output when it finishes. A static site
has no listing of a folder, and the pages link to folders (``input/``, ``expected/``, the folder
of an example), so each folder gets an ``index.html``.

**The page of an example.** The folder of an example (one that holds a ``script.vtl``) gets a page
of its own when ``operator_map/example_pages.json``, which ``generate.py`` writes, has an entry for
it: a link back to the example on the page of its operator, its title, status and the marks of the
engines, its notes, and then each of its files under its path, with a link to download it. The
notes are not a file of the page: ``notes.rst`` is rendered with docutils, and ``notes.md`` (the
chapters that have not moved to ``notes.rst`` yet) is shown as text. A CSV file is a table, and any
other file text. The page works on its own, when it is shared: it links the CSS of the map only for
the colours of the status. This script imports nothing of the tool, because ``conf.py`` imports it
for every version of the docs: ``STATUSES`` is a copy of what the pages of the operators say, and a
check keeps it in step.

**The listing of a folder.** Every other folder, and the folder of an example that has no entry in
``example_pages.json`` (a version of the docs that predates it, or a stale copy), gets a listing:
its folders first, then its files, with a link upwards. A static host may serve a ``.vtl`` or a
``.csv`` file as a download rather than as text, so the listing of an example folder, and of each
folder in it, also shows the text of every file of the folder and of its subfolders: the script,
the inputs, the expected outputs, the value domains, the external routines and the notes, each
under its path. The folders above the examples show no file: they would show every example below
them.

A file over 64 KB, or that is not UTF-8 text, is not shown, in a page or a listing: it is linked
instead. Everything that comes from a file or from ``example_pages.json`` is escaped, and every
URL quoted.

``conf.py`` calls ``copy_examples`` when a build has finished. A version of the docs that has no
operator map has no ``operator_map/examples`` folder, and nothing is copied.
"""

from __future__ import annotations

import csv
import html
import io
import json
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import quote

EXAMPLES = Path("operator_map") / "examples"
PAGES = Path("operator_map") / "example_pages.json"
_INDEX = "index.html"
_SCRIPT = "script.vtl"  # what makes a folder the folder of an example
_MAX_SHOWN = 64 * 1024  # bytes: a larger file is linked, not shown
# The files of an example in the order a reader takes them: the script, its inputs, what it
# expects, the other inputs of ``run`` and the notes. Anything else comes after them.
_ORDER = ("script.vtl", "input", "expected", "value_domains", "external_routines", "notes.md")
_NOTES = ("notes.rst", "notes.md")  # on the page of an example as text, not as files
# What the entry of an example in ``example_pages.json`` has: text, each of them.
_FIELDS = ("title", "status", "pandas", "duckdb", "operator_page", "anchor")
# The statuses of an example: its emoji (``EMOJI`` of ``tool.status``) and what it means
# (``_MEANING`` of ``tool.render``), as the legend of the pages of the operators says.
STATUSES = {
    "ok": ("🟢", "all ok"),
    "partial": ("🟡", "partially implemented"),
    "refused": ("🛡️", "refused with an error"),
    "missing": ("🔴", "not implemented or crash"),
    "diverge": ("🔵", "engines differ"),
}
# The page of an example stands on its own; it takes only the colours of the chip from the CSS of
# the map. A table keeps the spaces of its cells, as the data are in the file, and an inline
# literal of the notes (docutils writes it as ``<span class="docutils literal">``) is code.
_STYLE = """
body { font-family: Lato, Helvetica, Arial, sans-serif; max-width: 900px; margin: 24px auto;
  padding: 0 16px; color: #222; }
h1 { font-size: 22px; margin: 4px 0; }
h2 { font-size: 16px; margin: 18px 0 6px; }
a { color: #2980b9; }
.crumb { font-size: 13px; }
.dl { font-weight: normal; text-decoration: none; }
pre { background: #f5f5f5; border: 1px solid #ddd; padding: 8px 10px; overflow: auto; }
.data { overflow-x: auto; }
table { border-collapse: collapse; margin: 4px 0 12px; }
td, th { border: 1px solid #ccc; padding: 3px 8px; white-space: pre-wrap; }
th { background: #eee; }
.empty { color: #666; margin: 4px 0 12px; }
.notes { border-left: 4px solid #2980b9; background: #f7fbff; padding: 4px 12px; margin: 10px 0; }
.notes pre { background: none; border: none; padding: 0; white-space: pre-wrap; }
.notes .literal { font-family: Menlo, Consolas, "Liberation Mono", monospace; font-size: 90%; }
"""


def copy_examples(srcdir: Path, outdir: Path) -> bool:
    """Copy ``<srcdir>/operator_map/examples`` to ``<outdir>/operator_map/examples``.

    Every folder of the copy that has no ``index.html`` of its own gets one: the page of an
    example that ``example_pages.json`` has an entry for, a listing for any other folder. Returns
    whether there was anything to copy. Raises ``ValueError``, before it copies anything, when
    ``example_pages.json`` is not what ``generate.py`` writes.
    """
    source = srcdir / EXAMPLES
    if not source.is_dir():
        return False
    pages = _read_pages(srcdir / PAGES)
    target = outdir / EXAMPLES
    shutil.copytree(source, target, dirs_exist_ok=True)
    examples = {script.parent for script in source.rglob(_SCRIPT) if script.is_file()}
    for folder in [source, *sorted(path for path in source.rglob("*") if path.is_dir())]:
        if not (folder / _INDEX).exists():
            rel = folder.relative_to(source)
            page = pages.get(rel.as_posix()) if folder in examples else None
            if page is not None:
                text = _example_page(folder, rel, page)
            else:
                in_example = any(each in examples for each in (folder, *folder.parents))
                text = _listing(folder, source, in_example)
            (target / rel / _INDEX).write_text(text, encoding="utf-8")
    return True


def _read_pages(path: Path) -> dict[str, dict[str, str]]:
    """The entries of ``example_pages.json``, by the path of the example under ``examples``.

    A version of the docs that has no such file has no page of an example: there are no entries.
    A file that ``generate.py`` did not write, or an entry that lacks one of ``_FIELDS`` or has a
    status that is not in ``STATUSES``, raises ``ValueError`` that says where.
    """
    try:
        pages = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except ValueError as exc:
        raise ValueError(f"{path}: not JSON ({exc})") from exc
    if not isinstance(pages, dict):
        raise ValueError(f"{path}: expected an object with an entry for each example")
    for key, page in pages.items():
        if not isinstance(page, dict):
            raise ValueError(f"{path}: {key}: expected an object")
        for name in _FIELDS:
            if not isinstance(page.get(name), str):
                raise ValueError(f"{path}: {key}: {name} must be text")
        if page["status"] not in STATUSES:
            raise ValueError(f"{path}: {key}: unknown status {page['status']!r}")
    return pages


# --- The page of an example ---


def _example_page(folder: Path, rel: Path, page: dict[str, str]) -> str:
    """The page of the folder of an example: where it is, what it is, its notes and its files.

    ``rel`` is the folder under ``examples``, and ``page`` its entry of ``example_pages.json``.
    The page is ``operator_map/examples/<rel>/index.html``, so the folder of the map is one
    folder above the examples, and the root of the site one more.
    """
    up = "../" * (len(rel.parts) + 1)
    title = html.escape(page["title"])
    form = html.escape("/".join(rel.parts[:-1]))
    operator = html.escape(PurePosixPath(page["operator_page"]).stem)
    back = f"{up}{quote(page['operator_page'])}#{quote(page['anchor'])}"
    emoji, meaning = STATUSES[page["status"]]
    chip = f'<span class="om-chip {page["status"]}">{emoji} {meaning}</span>'
    marks = f"pandas {_mark(page['pandas'])} · DuckDB {_mark(page['duckdb'])}"
    files = "".join(_file(path, folder) for path in _example_files(folder))
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        '<head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title} · {form}</title>\n"
        f'<link rel="stylesheet" href="{up}../_static/operator_map.css">\n'
        f"<style>{_STYLE}</style></head>\n"
        "<body>\n"
        f'<p class="crumb"><a href="{back}">← {operator} operator page</a>'
        f" · {html.escape(rel.as_posix())}</p>\n"
        f"<h1>{title} {chip}</h1>\n"
        f'<p class="marks">{marks}</p>\n'
        f"{_notes(folder)}{files}"
        "</body>\n</html>\n"
    )


def _mark(text: str) -> str:
    """A mark as ``tool.status.mark`` writes it, ``✅`` or ``🛡 <code>``: the code in ``<code>``."""
    symbol, _, rest = text.partition(" ")
    return html.escape(symbol) + (f" <code>{html.escape(rest)}</code>" if rest else "")


def _notes(folder: Path) -> str:
    """The notes of an example as HTML: ``notes.rst`` rendered, else ``notes.md`` as text.

    The text of ``notes.md`` is preformatted and escaped: it is markdown, which is not read. An
    example with no notes has no notes section.
    """
    rst, md = folder / "notes.rst", folder / "notes.md"
    if rst.is_file():
        # Imported here: ``conf.py`` imports this script for every version of the docs, and the
        # checks of the map load it where docutils may be missing. Sphinx brings docutils.
        from docutils.core import publish_parts

        parts = publish_parts(
            rst.read_text(encoding="utf-8-sig"),
            writer_name="html5",
            settings_overrides={"docinfo_xform": False, "report_level": 5},
        )
        return f'<div class="notes">\n{parts["body"]}</div>\n'
    if md.is_file():
        text = html.escape(md.read_text(encoding="utf-8-sig"))
        return f'<div class="notes"><pre>{text}</pre></div>\n'
    return ""


def _example_files(folder: Path) -> list[Path]:
    """The files of an example folder that its page shows, in the order of ``_ORDER``.

    The notes are rendered at the top of the page, not shown as files, and a hidden file is not
    part of an example.
    """
    names = [path.relative_to(folder) for path in folder.rglob("*") if path.is_file()]
    kept = [
        name
        for name in names
        if name.as_posix() not in _NOTES and not any(part.startswith(".") for part in name.parts)
    ]
    return [folder / name for name in sorted(kept, key=lambda name: _reading_order(name.parts))]


def _file(path: Path, folder: Path) -> str:
    """A file on the page of an example: its path, a link to download it, and what it holds.

    A CSV file is a table and any other file text. A file that is not shown says why.
    """
    name = path.relative_to(folder).as_posix()
    label = html.escape(name)
    link = (
        f'<a class="dl" href="{quote(name)}" download title="Download {label}" '
        f'aria-label="Download {label}">⇩</a>'
    )
    heading = f"<h2>{label} {link}</h2>\n"
    text, reason = _read(path)
    if reason:
        return f"{heading}<p>Not shown here: {reason}.</p>\n"
    if path.suffix.lower() == ".csv":
        return heading + _table(text)
    return f"{heading}<pre>{html.escape(text)}</pre>\n"


def _table(text: str) -> str:
    """CSV text as a table, as the file writes it: the first row is the header, the rest data.

    A file with a header only is a table with no row, which it says; an empty file has no table.
    """
    rows = list(csv.reader(io.StringIO(text.removeprefix("\ufeff"), newline="")))
    if not rows:
        return '<p class="empty">(empty file)</p>\n'
    head = "".join(f"<th>{_cell(cell)}</th>" for cell in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{_cell(cell)}</td>" for cell in row) + "</tr>" for row in rows[1:]
    )
    note = "" if body else '<p class="empty">(no data points)</p>\n'
    return (
        f'<div class="data"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>\n{note}"
    )


def _cell(text: str) -> str:
    """A cell as HTML on one line: escaped, with its line breaks as character references."""
    return html.escape(text).replace("\r", "&#13;").replace("\n", "&#10;")


# --- The listing of a folder ---


def _listing(folder: Path, root: Path, show_files: bool) -> str:
    """The page that lists ``folder``: its folders first, then its files, with a link upwards.

    With ``show_files``, the text of every file of the folder and of its subfolders follows.
    """
    name = "/".join(("examples", *folder.relative_to(root).parts))
    entries = sorted(folder.iterdir(), key=lambda path: (not path.is_dir(), path.name))
    items = [] if folder == root else ['<li><a href="../">../</a></li>']
    for entry in entries:
        slash = "/" if entry.is_dir() else ""
        items.append(
            f'<li><a href="{quote(entry.name)}{slash}">{html.escape(entry.name)}{slash}</a></li>'
        )
    sections: list[str] = []
    if show_files:
        files = sorted(
            (path for path in folder.rglob("*") if path.is_file()),
            key=lambda path: _reading_order(path.relative_to(folder).parts),
        )
        sections = [_shown(path, folder) for path in files]
    title = html.escape(name)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        f'<head><meta charset="utf-8"><title>{title}</title></head>\n'
        f"<body><h1>{title}/</h1>\n<ul>\n"
        + "\n".join(items)
        + "\n</ul>\n"
        + "".join(sections)
        + "</body>\n</html>\n"
    )


def _reading_order(parts: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    """Where a file of an example comes: by ``_ORDER`` of its first part, then by its path."""
    rank = _ORDER.index(parts[0]) if parts[0] in _ORDER else len(_ORDER)
    return rank, parts


def _read(path: Path) -> tuple[str, str]:
    """The text of a file and no reason, or no text and the reason that it is not shown.

    A file of more than ``_MAX_SHOWN`` bytes, or that is not UTF-8 text, is not shown.
    """
    size = path.stat().st_size
    if size > _MAX_SHOWN:
        return "", f"{size} bytes, more than {_MAX_SHOWN}"
    try:
        return path.read_text(encoding="utf-8"), ""
    except UnicodeDecodeError:
        return "", "it is not UTF-8 text"


def _shown(path: Path, folder: Path) -> str:
    """A file in the page of ``folder``: its path as a heading, then its text, or a link."""
    name = path.relative_to(folder).as_posix()
    heading = f"<h2>{html.escape(name)}</h2>\n"
    text, reason = _read(path)
    if reason:
        link = f'<a href="{quote(name)}">Open the file</a>.'
        return f"{heading}<p>Not shown here: {reason}. {link}</p>\n"
    return f"{heading}<pre>{html.escape(text)}</pre>\n"
