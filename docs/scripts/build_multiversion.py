#!/usr/bin/env python3
"""Build multi-version documentation by checking out each ref and running sphinx-build.

Reads the ordered list of refs to build from ``docs/_versions.json`` (produced by
``configure_doc_versions.py``) and runs ``sphinx-build`` once per ref.

For historical tags, the current branch's ``docs/conf.py``, ``docs/_templates/``, and
``docs/scripts/`` are overlaid on top of the worktree before building so legacy refs
build without depending on whatever extension stack they shipped with at the time.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VERSIONS_JSON_PATH = REPO_ROOT / "docs" / "_versions.json"

# Paths relative to repo root copied from the current checkout into each historical
# worktree. Limiting the overlay to conf + templates + scripts keeps each version's
# RST content and _static/ untouched.
OVERLAY_PATHS = (
    "docs/conf.py",
    "docs/_templates",
    "docs/scripts",
)


def load_versions() -> tuple[list[str], Optional[str]]:
    """Read the resolved version list and latest stable from _versions.json."""
    if not VERSIONS_JSON_PATH.exists():
        raise SystemExit(
            f"Missing {VERSIONS_JSON_PATH.relative_to(REPO_ROOT)}. "
            f"Run docs/scripts/configure_doc_versions.py first."
        )
    data = json.loads(VERSIONS_JSON_PATH.read_text(encoding="utf-8"))
    return list(data["versions"]), data.get("latest_stable")


def overlay_current_docs(worktree: Path) -> None:
    """Copy the current checkout's docs/conf.py, _templates, and scripts over `worktree`."""
    for rel in OVERLAY_PATHS:
        src = REPO_ROOT / rel
        dst = worktree / rel
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def build_one_version(
    ref: str,
    out_dir: Path,
    versions: list[str],
    latest_stable: Optional[str],
) -> None:
    """Check out `ref` into a temp worktree, overlay current docs, and run sphinx-build."""
    with tempfile.TemporaryDirectory(prefix=f"vtlengine-docs-{ref.replace('/', '_')}-") as tmp:
        worktree = Path(tmp) / "wt"
        subprocess.run(  # noqa: S603
            ["git", "worktree", "add", "--detach", str(worktree), ref],  # noqa: S607
            cwd=REPO_ROOT,
            check=True,
        )
        try:
            overlay_current_docs(worktree)

            env = os.environ.copy()
            env["VTLENGINE_DOCS_CURRENT_VERSION"] = ref
            env["VTLENGINE_DOCS_VERSIONS_JSON"] = json.dumps(versions)
            if latest_stable:
                env["VTLENGINE_DOCS_LATEST_STABLE"] = latest_stable
            else:
                env.pop("VTLENGINE_DOCS_LATEST_STABLE", None)

            subprocess.run(  # noqa: S603
                [  # noqa: S607
                    "sphinx-build",
                    "-b",
                    "html",
                    str(worktree / "docs"),
                    str(out_dir / ref),
                ],
                env=env,
                check=True,
            )
        finally:
            subprocess.run(  # noqa: S603
                ["git", "worktree", "remove", "--force", str(worktree)],  # noqa: S607
                cwd=REPO_ROOT,
                check=False,
            )


def main() -> int:
    """Build every ref in docs/_versions.json into <output_dir>/<ref>/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Top-level output directory (e.g. _site).",
    )
    args = parser.parse_args()

    versions, latest_stable = load_versions()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for ref in versions:
        print(f"\n=== Building {ref} -> {args.output_dir / ref} ===", flush=True)
        build_one_version(ref, args.output_dir, versions, latest_stable)

    print("\nAll versions built successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
