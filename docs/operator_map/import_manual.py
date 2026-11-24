"""Write the index of the VTL 2.2 manual, or import the manual examples of a chapter.

    python docs/operator_map/import_manual.py --spec SPEC --index
    python docs/operator_map/import_manual.py --spec SPEC --chapter <id> [--force]

SPEC is ``v2.2/docs/reference_manual`` in a clone of https://github.com/sdmx-twg/vtl, at the
commit the map is pinned to. ``--index`` writes ``catalogue/manual_index.py``, which lists the
pages and the examples of the manual for the coverage checks of the catalogue. ``--chapter``
imports the manual examples of a chapter of the catalogue into ``examples/`` and prints the
warnings that need a hand. An example folder that has a script, inputs or expected outputs is
left as it is unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from tool.importer import ManualImportError, build_index_source, import_chapter, list_manual
from tool.loader import load_catalogue
from tool.model import DL_PAGES

_ROOT = Path(__file__).resolve().parent
_INDEX = _ROOT / "catalogue" / "manual_index.py"
_EXAMPLES = _ROOT / "examples"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", type=Path, required=True, help="v2.2/docs/reference_manual")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--index", action="store_true", help="write catalogue/manual_index.py")
    action.add_argument("--chapter", metavar="ID", help="import the manual examples of a chapter")
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --chapter: replace the script, inputs and expected files that exist",
    )
    parser.add_argument(
        "--commit", help="with --index: the commit of the manual (default: HEAD of its clone)"
    )
    args = parser.parse_args(argv)
    if args.index and args.force:
        parser.error("--force goes with --chapter")
    if args.chapter is not None and args.commit is not None:
        parser.error("--commit goes with --index")
    try:
        if args.index:
            return _write_index(args.spec, args.commit)
        return _import_chapter(args.spec, args.chapter, args.force)
    except ManualImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _write_index(spec: Path, commit: str | None) -> int:
    commit = commit or _git_commit(spec)
    source = build_index_source(spec, commit)
    _INDEX.write_bytes(source.encode("utf-8"))
    pages, examples = list_manual(spec)
    print(
        f"wrote {_INDEX}: {len(pages)} pages and {len(DL_PAGES)} VTL-DL pages, "
        f"{len(examples)} examples, manual at {commit}"
    )
    return 0


def _git_commit(spec: Path) -> str:
    """The commit that the clone of the manual is at."""
    git = shutil.which("git")
    if git is None:
        raise ManualImportError("git was not found: give the commit of the manual with --commit")
    done = subprocess.run(  # noqa: S603
        [git, "-C", str(spec), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if done.returncode != 0:
        raise ManualImportError(
            f"{spec} is not in a git clone ({done.stderr.strip()}): "
            "give the commit of the manual with --commit"
        )
    return done.stdout.strip()


def _import_chapter(spec: Path, chapter_id: str, force: bool) -> int:
    chapters = load_catalogue().chapters
    chapter = next((item for item in chapters if item.id == chapter_id), None)
    if chapter is None:
        known = ", ".join(item.id for item in chapters) or "none yet"
        raise ManualImportError(f"no chapter '{chapter_id}' in the catalogue (known: {known})")
    report = import_chapter(chapter, spec, _EXAMPLES, force)
    for warning in report.warnings:
        print(f"warning: {warning}")
    print(f"{chapter.id}: {report.imported} imported, {len(report.skipped)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
