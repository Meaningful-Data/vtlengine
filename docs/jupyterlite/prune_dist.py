"""Trim the served Pyodide distribution to what the demo can reach.

``jupyter lite build --pyodide=<tarball>`` copies the whole Pyodide distribution
into ``static/pyodide``: every package the Pyodide project builds (polars, scipy,
opencv...), the unvendored test suites of those packages (``*-tests.zip``),
CPython's own ``test`` package and Pyodide's self-test fixtures. That is ~380 MB,
of which ``import vtlengine`` and the kernel can reach ~60 MB. A browser only
downloads what a notebook imports, so the surplus costs visitors nothing, but it
is ~90% of the Pages artifact (GitHub caps a published site at 1 GB).

Run *after* ``patch_lock.py`` (the closure is computed on the patched lockfile)::

    python prune_dist.py <output>/static/pyodide <kernel-extension>/static/pypi

Kept:

* the dependency closure of the injected packages (``patch_lock.EXTRA``);
* the closure of what the kernel needs at boot: ``micropip`` (the worker loads it
  before anything else) and the requirements of the kernel's own wheels in
  ``static/pypi`` (``ipython`` and its stack);
* every non-package file: the runtime itself (``pyodide.asm.wasm``,
  ``python_stdlib.zip``...) is never touched.

The closure follows both dependency graphs the demo loads packages through: the
lockfile ``depends`` (what ``import x`` auto-loads) and the wheels' own METADATA
requirements evaluated for Pyodide, extras included (what ``micropip`` resolves
when the kernel installs itself at boot: ``ipython`` pulls ``jedi`` that way,
not through ``depends``).

Removed: every other lockfile package, whose entry is dropped from
``pyodide-lock.json`` too, and the package files the lockfile never referenced.

Trade-off: ``%pip install`` of a *compiled* package that was pruned (scipy,
polars...) no longer works in the demo; pure-Python packages still come from PyPI.
"""

import json
import re
import sys
import zipfile
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from patch_lock import EXTRA

# Loaded by the kernel worker before anything else (`loadPackage(["micropip"])`).
KERNEL_ROOTS = ["micropip"]
# The Python standard library is a core file, not a package: never a prune candidate.
RUNTIME_ZIPS = {"python_stdlib.zip"}

NO_EXTRAS: frozenset[str] = frozenset()


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def marker_environment(lock: dict) -> dict[str, str]:
    """The PEP 508 environment of the served Pyodide, as micropip evaluates markers."""
    full_version = lock["info"]["python"]
    return {
        "implementation_name": "cpython",
        "implementation_version": full_version,
        "os_name": "posix",
        "platform_machine": lock["info"].get("arch", "wasm32"),
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": "Emscripten",
        "platform_version": "",
        "python_full_version": full_version,
        "python_version": ".".join(full_version.split(".")[:2]),
        "sys_platform": "emscripten",
    }


def wheel_requirements(whl: Path, extras: frozenset[str], env: dict[str, str]) -> list[Requirement]:
    """The wheel's METADATA requirements that apply on Pyodide with ``extras`` requested."""
    wanted = []
    with zipfile.ZipFile(whl) as zf:
        metadata = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        for line in zf.read(metadata).decode().splitlines():
            if not line.startswith("Requires-Dist:"):
                continue
            try:
                requirement = Requirement(line.split(":", 1)[1].strip())
            except InvalidRequirement:
                continue
            if requirement.marker is None or any(
                requirement.marker.evaluate({**env, "extra": extra}) for extra in extras or {""}
            ):
                wanted.append(requirement)
    return wanted


def kernel_requirements(pypi: Path, env: dict[str, str]) -> list[Requirement]:
    """What the kernel's own wheels (static/pypi) require when micropip installs them."""
    wheels = sorted(pypi.glob("*.whl"))
    if not wheels:
        # Without them the IPython stack would be pruned and the kernel would not boot.
        raise SystemExit(f"ERROR: no kernel wheels in {pypi}: did the kernel move them?")
    return [req for whl in wheels for req in wheel_requirements(whl, NO_EXTRAS, env)]


def closure(dist: Path, lock: dict, roots: list) -> tuple[set[str], set[str]]:
    """Lockfile keys reachable from ``roots`` (names or Requirements), and unknown names."""
    packages = lock["packages"]
    env = marker_environment(lock)
    by_norm = {norm(name): name for name in packages}
    seen: dict[str, set[str]] = {}  # kept package -> extras already expanded
    unknown: set[str] = set()
    todo: list[tuple[str, frozenset[str]]] = [
        (root.name, frozenset(root.extras)) if isinstance(root, Requirement) else (root, NO_EXTRAS)
        for root in roots
    ]
    while todo:
        wanted, extras = todo.pop()
        name = by_norm.get(norm(wanted))
        if name is None:
            unknown.add(wanted)
            continue
        if name in seen and extras <= seen[name]:
            continue
        first_visit = name not in seen
        seen.setdefault(name, set()).update(extras)
        if first_visit:
            todo.extend((dep, NO_EXTRAS) for dep in packages[name]["depends"])
        wheel = dist / packages[name]["file_name"]
        if wheel.suffix == ".whl" and wheel.exists():
            todo.extend(
                (req.name, frozenset(req.extras)) for req in wheel_requirements(wheel, extras, env)
            )
    return set(seen), unknown


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


def main() -> None:
    dist, pypi = Path(sys.argv[1]), Path(sys.argv[2])
    lock_path = dist / "pyodide-lock.json"
    lock = json.loads(lock_path.read_text())
    packages = lock["packages"]

    missing = [pkg for pkg in EXTRA if pkg not in packages]
    if missing:
        raise SystemExit(f"ERROR: {missing} not in {lock_path}: run patch_lock.py first")

    roots = [*EXTRA, *KERNEL_ROOTS, *kernel_requirements(pypi, marker_environment(lock))]
    keep, unknown = closure(dist, lock, roots)
    kept_files = {packages[name]["file_name"] for name in keep}

    before, n_files_before = dir_size(dist), len(list(dist.iterdir()))
    removed_bytes = 0
    for file in sorted(dist.iterdir()):
        if not file.is_file():
            continue
        # A `.whl.metadata` sidecar follows its wheel.
        artifact = file.name.removesuffix(".metadata")
        is_package_file = artifact.endswith((".whl", ".zip")) and artifact not in RUNTIME_ZIPS
        if is_package_file and artifact not in kept_files:
            removed_bytes += file.stat().st_size
            file.unlink()

    lock["packages"] = {name: packages[name] for name in sorted(packages) if name in keep}
    lock_path.write_text(json.dumps(lock))

    after, n_files_after = dir_size(dist), len(list(dist.iterdir()))
    mb = 1024 * 1024
    print(f"  kept {len(keep)} of {len(packages)} lockfile packages: {sorted(keep)}")
    print(f"  requirements not in the lockfile (kernel wheels / PyPI): {sorted(unknown)}")
    print(
        f"  static/pyodide: {n_files_before} files, {before / mb:.1f} MB -> "
        f"{n_files_after} files, {after / mb:.1f} MB (removed {removed_bytes / mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
