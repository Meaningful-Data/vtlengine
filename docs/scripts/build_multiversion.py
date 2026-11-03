#!/usr/bin/env python3
"""Build multi-version documentation by checking out each ref and running sphinx-build.

Reads the ordered list of refs to build from ``docs/_versions.json`` (produced by
``configure_doc_versions.py``) and runs ``sphinx-build`` once per ref.

For historical tags, the current branch's ``docs/conf.py``, ``docs/_templates/``, and
``docs/scripts/`` are overlaid on top of the worktree before building so legacy refs
build without depending on whatever extension stack they shipped with at the time.

With ``VTLENGINE_DOCS_REGENERATE_MAP=1`` (as ``docs.yml`` sets it), a version that has the
operator map (``docs/operator_map/generate.py``) gets its map regenerated against its own engine
before it is built: the worktree's source, with the compiled parser of the checkout when the
parser sources are the same, else the one of the version's wheel on PyPI. Without it, as in a
local build, the versions keep their committed maps: a regeneration runs every example of the
map, which takes minutes and memory. A map that cannot be regenerated never stops the build:
the version is built from its committed pages, and the reason is printed and added to the
summary of the CI job (``GITHUB_STEP_SUMMARY``), where the generator also writes its report.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
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

# The operator map of a version: its generator, and where its runs import the compiled parser.
MAP_GENERATOR = Path("docs/operator_map/generate.py")
PARSER_DIR = Path("src/vtlengine/AST/Grammar/_cpp_parser")
PARSER_SOURCES = ("src/vtlengine/AST/Grammar/_cpp_parser", "CMakeLists.txt")
PARSER_PREFIX = "vtl_cpp_parser"
MAP_TIMEOUT_S = 3600  # a run of every example takes about 10 minutes
REGENERATE_MAP = "VTLENGINE_DOCS_REGENERATE_MAP"  # "1" to regenerate the maps
_RELEASE_TAG = re.compile(r"v(\d+\.\d+\.\d+(?:rc\d+)?)")


class MapError(Exception):
    """Why the operator map of a version cannot be regenerated."""


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


def resolve_worktree_ref(ref: str) -> str:
    """Return a ref name that ``git worktree add`` can resolve.

    When the CI checkout is a detached HEAD at a tag, branch names like ``main``
    only exist as remote-tracking refs (``origin/main``); ``git worktree add``
    does not DWIM a bare branch name to its remote counterpart. Tries the ref
    as-is first, then falls back to ``origin/<ref>``.
    """
    for candidate in (ref, f"origin/{ref}"):
        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
        )
        if result.returncode == 0:
            return candidate
    raise SystemExit(f"Cannot resolve git ref {ref!r} (tried {ref!r} and {f'origin/{ref}'!r})")


def wheel_version(ref: str) -> Optional[str]:
    """The version on PyPI of a release tag (``v1.9.7`` is ``1.9.7``); None for a branch."""
    found = _RELEASE_TAG.fullmatch(ref)
    return found.group(1) if found else None


def install_parser(worktree: Path, ref: str) -> str:
    """Put the compiled parser of ``ref`` in the worktree's src/, where the map's runs import it.

    It is the checkout's own when the worktree has the same parser sources (the same commit, or
    a version whose grammar did not change), else the one of the version's wheel on PyPI, for
    the running Python. Returns where it came from; raises ``MapError`` when there is none.
    """
    target = worktree / PARSER_DIR
    if _same_parser_sources(worktree):
        binaries = sorted((REPO_ROOT / PARSER_DIR).glob(f"{PARSER_PREFIX}*"))
        if binaries:
            for binary in binaries:
                shutil.copy2(binary, target / binary.name)
            return "the checkout's build"
    version = wheel_version(ref)
    if version is None:
        raise MapError(
            f"{ref} has no release on PyPI and its parser sources are not the checkout's"
        )
    prefix = f"vtlengine/AST/Grammar/_cpp_parser/{PARSER_PREFIX}"
    with tempfile.TemporaryDirectory(prefix="vtlengine-wheel-") as tmp:
        done = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                f"vtlengine=={version}",
                "--no-deps",
                "--only-binary=:all:",
                "--dest",
                tmp,
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        wheels = sorted(Path(tmp).glob("vtlengine-*.whl"))
        if done.returncode != 0 or not wheels:
            reason = (done.stderr or "").strip().splitlines()[-1:] or ["no wheel"]
            raise MapError(f"pip cannot download vtlengine=={version}: {reason[0]}")
        with zipfile.ZipFile(wheels[0]) as wheel:
            names = [name for name in wheel.namelist() if name.startswith(prefix)]
            if not names:
                raise MapError(f"the wheel {wheels[0].name} has no compiled parser")
            for name in names:
                (target / Path(name).name).write_bytes(wheel.read(name))
    return f"vtlengine=={version} from PyPI"


def _same_parser_sources(worktree: Path) -> bool:
    """Whether the worktree's parser sources are those of the checkout's HEAD."""
    head = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        return False
    diff = subprocess.run(  # noqa: S603
        ["git", "diff", "--quiet", "HEAD", head.stdout.strip(), "--", *PARSER_SOURCES],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
    )
    return diff.returncode == 0


def regenerate_operator_map(worktree: Path, ref: str) -> None:
    """Regenerate the operator map of ``ref`` in its worktree, against its own engine.

    Only when ``VTLENGINE_DOCS_REGENERATE_MAP`` is ``1``; a version without the map is left
    alone. This never raises: when the parser cannot be had, or the generator fails, the version
    is built from its committed pages, and the reason is printed and appended to the summary of
    the CI job when there is one.
    """
    if os.environ.get(REGENERATE_MAP) != "1" or not (worktree / MAP_GENERATOR).is_file():
        return
    summary = os.environ.get("GITHUB_STEP_SUMMARY", "")
    try:
        source = install_parser(worktree, ref)
        print(f"operator map of {ref}: regenerating it, with the parser of {source}", flush=True)
        command = [sys.executable, str(worktree / MAP_GENERATOR)]
        if summary:
            command += ["--summary", summary]
        done = subprocess.run(command, cwd=worktree, timeout=MAP_TIMEOUT_S, check=False)  # noqa: S603
        if done.returncode != 0:
            raise MapError(f"the generator exited with {done.returncode}")
    except Exception as exc:  # the docs are built whatever happens to the map
        reason = "timed out" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)
        line = f"operator map of {ref}: not regenerated ({reason}); built from its committed pages"
        print(line, flush=True)
        if summary:
            try:
                with open(summary, "a", encoding="utf-8") as report:
                    report.write(f"\n{line}\n")
            except OSError as error:
                print(f"cannot append to {summary}: {error}", flush=True)


def build_one_version(
    ref: str,
    out_dir: Path,
    versions: list[str],
    latest_stable: Optional[str],
) -> None:
    """Check out `ref` into a temp worktree, overlay current docs, and run sphinx-build."""
    resolved_ref = resolve_worktree_ref(ref)
    with tempfile.TemporaryDirectory(prefix=f"vtlengine-docs-{ref.replace('/', '_')}-") as tmp:
        worktree = Path(tmp) / "wt"
        subprocess.run(  # noqa: S603
            ["git", "worktree", "add", "--detach", str(worktree), resolved_ref],  # noqa: S607
            cwd=REPO_ROOT,
            check=True,
        )
        try:
            overlay_current_docs(worktree)
            regenerate_operator_map(worktree, ref)

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
