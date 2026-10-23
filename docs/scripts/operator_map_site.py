"""Serve the example files of the operator map next to its pages in the built HTML site.

The pages of ``docs/operator_map`` link, relative to themselves, to ``examples/...``: the scripts,
the input and expected data, the notes, and the folders that hold them. Those files are not
sources of Sphinx, so the build copies the folder to its output when it finishes. A static site
has no listing of a folder, and the pages link to folders (``input/``, ``expected/``, the folder
of an example), so each folder gets an ``index.html`` that lists it.

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
    for folder in [source, *sorted(path for path in source.rglob("*") if path.is_dir())]:
        if not (folder / _INDEX).exists():
            listing = target / folder.relative_to(source) / _INDEX
            listing.write_text(_listing(folder, source), encoding="utf-8")
    return True


def _listing(folder: Path, root: Path) -> str:
    """The page that lists ``folder``: its folders first, then its files, with a link upwards."""
    name = "/".join(("examples", *folder.relative_to(root).parts))
    entries = sorted(folder.iterdir(), key=lambda path: (not path.is_dir(), path.name))
    items = [] if folder == root else ['<li><a href="../">../</a></li>']
    for entry in entries:
        slash = "/" if entry.is_dir() else ""
        items.append(
            f'<li><a href="{quote(entry.name)}{slash}">{html.escape(entry.name)}{slash}</a></li>'
        )
    title = html.escape(name)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        f'<head><meta charset="utf-8"><title>{title}</title></head>\n'
        f"<body><h1>{title}/</h1>\n<ul>\n" + "\n".join(items) + "\n</ul></body>\n</html>\n"
    )
