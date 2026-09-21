"""Merge micropip's resolution of the vtlengine wheel into the served Pyodide lockfile, so
that ``import vtlengine`` auto-loads everything in JupyterLite with no ``%pip``/``piplite``
step.

Run *after* ``jupyter lite build`` and ``resolve_wheels.mjs``::

    python patch_lock.py <output>/static/pyodide <micropip-lock.json> <vtlengine wheel>

``<micropip-lock.json>`` is ``micropip.freeze()`` after installing the wheel in that very
distribution (``resolve_wheels.mjs``), so its entries and their ``depends`` are what
micropip resolved. Whatever it added to, or swapped in, the distribution's lockfile
(vtlengine and the dependencies Pyodide does not ship) goes into the served
``pyodide-lock.json``, with the wheels downloaded next to it (sha256 checked) so the demo
stays self-contained, and with ``imports`` filled from the wheel contents where micropip
left them empty: ``imports`` is what drives JupyterLite's import-triggered auto-load.
Everything else the distribution ships is left as is, except for the three deviations at
the end of ``main`` (see README.md).
"""

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path


def top_level_imports(whl: Path) -> list[str]:
    """The top-level packages and modules a wheel installs, what ``import x`` looks up."""
    names: set[str] = set()
    with zipfile.ZipFile(whl) as zf:
        for member in zf.namelist():
            head, _, tail = member.partition("/")
            if head.endswith((".dist-info", ".data")):
                continue
            if tail:
                names.add(head)
            elif head.endswith(".py"):
                names.add(head[:-3])
    return sorted(names)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: python patch_lock.py <static/pyodide> <micropip-lock.json> <whl>")
    dist, resolution, wheel = (Path(arg) for arg in sys.argv[1:])
    lock_path = dist / "pyodide-lock.json"
    lock = json.loads(lock_path.read_text())
    packages = lock["packages"]
    resolved = json.loads(resolution.read_text())["packages"]
    if "vtlengine" not in resolved:
        raise SystemExit(
            f"ERROR: {resolution} does not contain vtlengine: did micropip install it?"
        )

    for name, entry in sorted(resolved.items()):
        # micropip records where it loaded each wheel from: the distribution's own files as
        # absolute paths of that directory, PyPI wheels as URLs, ours as its emfs: path.
        source = entry["file_name"]
        target = dist / source.rsplit("/", 1)[-1]
        if name in packages and packages[name]["file_name"] == target.name:
            continue  # shipped by the distribution and taken from it: left untouched
        if source.startswith("emfs:"):
            if target.name != wheel.name:
                raise SystemExit(f"ERROR: micropip installed {target.name}, not {wheel.name}")
            target.write_bytes(wheel.read_bytes())
        else:
            if not source.startswith("https://"):
                raise SystemExit(f"ERROR: unexpected wheel location for {name}: {source}")
            with urllib.request.urlopen(source) as response:  # noqa: S310  (https only)
                target.write_bytes(response.read())
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit(f"ERROR: {target.name} does not match the sha256 micropip recorded")
        entry["file_name"] = target.name
        entry["imports"] = entry["imports"] or top_level_imports(target)
        entry.setdefault("package_type", "package")
        entry.setdefault("unvendored_tests", False)
        verb = "replaces the distribution's" if name in packages else "from"
        origin = "the wheel" if source.startswith("emfs:") else "PyPI"
        packages[name] = entry
        depends = " ".join(entry["depends"]) or "-"
        print(f"  + {name} {entry['version']} {verb} {origin}, depends: {depends}")

    # Three deliberate differences from a plain micropip install on stock Pyodide (README.md).
    # JupyterLite loads packages per `import`, while micropip.install loads the whole set at
    # once, so a lockfile entry has to bring what its package needs at runtime.
    # pysdmx reads and writes data with pandas and pyarrow (its `data` extra). vtlengine asks
    # for `pysdmx[xml]` and brings both itself, so micropip records neither under pysdmx and
    # a notebook importing pysdmx before vtlengine would get it without them.
    for name in ("pandas", "pyarrow"):
        if name not in packages["pysdmx"]["depends"]:
            packages["pysdmx"]["depends"].append(name)
    # Pyodide's networkx recipe lists matplotlib (and networkx 2.x's decorator/setuptools)
    # as run dependencies; networkx 3.x declares none and vtlengine needs none of them.
    packages["networkx"]["depends"] = []
    # certifi ships in stock Pyodide but nothing auto-loads it; httpx imports it for HTTPS,
    # so a remote read_sdmx/run_sdmx(URL) fails without it. Local-file SDMX needs none of this.
    if "certifi" not in packages["httpx"]["depends"]:
        packages["httpx"]["depends"].append("certifi")

    lock_path.write_text(json.dumps(lock))
    print(f"patched {lock_path} ({len(packages)} packages)")


if __name__ == "__main__":
    main()
