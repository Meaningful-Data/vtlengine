"""Serve the example files of the operator map next to its pages in the built HTML site.

The pages of ``docs/operator_map`` link, relative to themselves, to ``examples/...``: the scripts,
the input and expected data, the notes, and the folders that hold them. Those files are not
sources of Sphinx, so the build copies the folder to its output when it finishes. A static site
has no listing of a folder, and the pages link to folders (``input/``, ``expected/``, the folder
of an example), so each folder gets an ``index.html`` that lists it.

A static host may serve a ``.vtl`` or a ``.csv`` file as a download rather than as text, so the
page of an example folder, and of each folder in it, also shows the text of every file of the
folder and of its subfolders: the script, the inputs, the expected outputs, the value domains,
the external routines and the notes, each under its path. A file over 64 KB, or that is not
UTF-8 text, is linked instead. The folders above the examples show no file: they would show
every example below them.

``conf.py`` calls ``copy_examples`` when a build has finished. A version of the docs that has no
operator map has no ``operator_map/examples`` folder, and nothing is copied.
"""

from __future__ import annotations

import html
import shutil
from pathlib import Path
from urllib.parse import quote

EXAMPLES = Path("operator_map") / "examples"
_INDEX = "index.html"
_SCRIPT = "script.vtl"  # what makes a folder the folder of an example
_MAX_SHOWN = 64 * 1024  # bytes: a larger file is linked, not shown
# The files of an example in the order a reader takes them: the script, its inputs, what it
# expects, the other inputs of ``run`` and the notes. Anything else comes after them.
_ORDER = ("script.vtl", "input", "expected", "value_domains", "external_routines", "notes.md")


def copy_examples(srcdir: Path, outdir: Path) -> bool:
    """Copy ``<srcdir>/operator_map/examples`` to ``<outdir>/operator_map/examples``.

    Every folder of the copy that has no ``index.html`` of its own gets one. Returns whether
    there was anything to copy.
    """
    source = srcdir / EXAMPLES
    if not source.is_dir():
        return False
    target = outdir / EXAMPLES
    shutil.copytree(source, target, dirs_exist_ok=True)
    examples = {script.parent for script in source.rglob(_SCRIPT) if script.is_file()}
    for folder in [source, *sorted(path for path in source.rglob("*") if path.is_dir())]:
        if not (folder / _INDEX).exists():
            listing = target / folder.relative_to(source) / _INDEX
            in_example = any(each in examples for each in (folder, *folder.parents))
            listing.write_text(_listing(folder, source, in_example), encoding="utf-8")
    return True


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


def _shown(path: Path, folder: Path) -> str:
    """A file in the page of ``folder``: its path as a heading, then its text, or a link."""
    name = path.relative_to(folder).as_posix()
    heading = f"<h2>{html.escape(name)}</h2>\n"
    link = f'<a href="{quote(name)}">Open the file</a>.'
    size = path.stat().st_size
    if size > _MAX_SHOWN:
        return f"{heading}<p>Not shown here: {size} bytes, more than {_MAX_SHOWN}. {link}</p>\n"
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"{heading}<p>Not shown here: it is not UTF-8 text. {link}</p>\n"
    return f"{heading}<pre>{html.escape(text)}</pre>\n"
