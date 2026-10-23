"""The child process of the runner: one example on one engine, one line of result.

``python -m tool.child <example folder> <engine> <repeat>`` loads the example, runs its script
with ``vtlengine.run`` and prints one line to stdout, ``RESULT:<json>``, then exits with 0. The
JSON has one of two shapes:

* ``{"ok": true, "datasets": [...], "scalars": [...], "stable": null | true | false}``: the
  datasets and scalars are ``dataclasses.asdict`` of ``ActualDataset`` and ``ActualScalar``, and
  ``stable`` says whether ``repeat`` runs gave the same outputs (null for a single run);
* ``{"ok": false, "error_type": ..., "error_code": ..., "error_message": ...}``.

Every failure is the second shape, an example that cannot be loaded as well as an error of the
engine, and the exit code is 0 either way. Only a process that dies, by a crash or by a kill of
the kernel, leaves no result line, and the parent reads that as a crash. The result line is the
only thing written to stdout: whatever else is printed goes to stderr.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, TextIO

import pandas as pd

from tool.compare import (
    ActualComponent,
    ActualDataset,
    ActualScalar,
    Output,
    normalize_role,
    normalize_type,
)
from tool.examples import ExampleFiles, load_example
from vtlengine import run
from vtlengine.Model import Dataset, Scalar

RESULT_PREFIX: Final = "RESULT:"
MAX_MESSAGE: Final = 300
ENGINES: Final = ("pandas", "duckdb")

# The code of a vtlengine error, such as ``1-1-1-1`` or ``2-3-6``: digits in two groups or more.
_CODE: Final = re.compile(r"\d+(-\d+)+", re.ASCII)


def serialise(results: Mapping[str, Any]) -> list[Output]:
    """The datasets and scalars that ``run`` returned, in its order, as the comparator takes them.

    A component keeps its role and type under their canonical names, and a data point keeps the
    order of the components. A value is its ``str``, whatever the engine stores: dates, time
    periods, durations and time intervals are plain texts there. A null (None, NaN, NaT or
    ``pd.NA``) is None. A dataset without data has no data points.
    """
    outputs: list[Output] = []
    for name, result in results.items():
        if isinstance(result, Dataset):
            outputs.append(_dataset(name, result))
        elif isinstance(result, Scalar):
            outputs.append(ActualScalar(name, _type_name(result.data_type), _text(result.value)))
        else:
            raise TypeError(
                f"{name}: a Dataset or a Scalar was expected, not {type(result).__name__}"
            )
    return outputs


def collect(run_once: Callable[[], list[Output]], repeat: int) -> tuple[list[Output], bool | None]:
    """The outputs of the first of ``repeat`` runs, and whether every run gave the same ones.

    A single run has nothing to agree with: the second item is then None. The runs stop at the
    first one that differs.
    """
    first = run_once()
    stable = all(run_once() == first for _ in range(repeat - 1)) if repeat > 1 else None
    return first, stable


def failure(exc: BaseException) -> dict[str, Any]:
    """The result line of a run that failed with ``exc``.

    vtlengine raises its errors with ``(text, code)`` as arguments, which ``str`` prints as the
    repr of that tuple. So the message is the first line of the first argument when that is a
    text, and the first line of ``str(exc)`` for any other exception. A line is stripped, and
    blank lines before it do not count. The code is the first argument that is a text of the
    form of a code, such as ``2-3-6``; it is None when there is none.
    """
    first = exc.args[0] if exc.args else None
    text: str = first if isinstance(first, str) else str(exc)
    lines = (line.strip() for line in text.splitlines())
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error_code": next(
            (arg for arg in exc.args if isinstance(arg, str) and _CODE.fullmatch(arg)), None
        ),
        "error_message": next((line for line in lines if line), "")[:MAX_MESSAGE],
    }


def main(argv: Sequence[str], out: TextIO) -> int:
    """Run the example that ``argv`` names and write the result line to ``out``; the exit code.

    Whatever goes wrong while the example loads and runs has an answer: the line of the failure,
    for an ``Exception`` as well as for a ``SystemExit`` that a library may raise. Wrong arguments
    are a mistake of the caller, which ``argparse`` reports on stderr before any run.
    """
    arguments = parse_arguments(argv)
    try:
        files = load_example(arguments.example)
        outputs, stable = collect(lambda: _run_once(files, arguments.engine), arguments.repeat)
        payload: dict[str, Any] = {
            "ok": True,
            "datasets": [asdict(item) for item in outputs if isinstance(item, ActualDataset)],
            "scalars": [asdict(item) for item in outputs if isinstance(item, ActualScalar)],
            "stable": stable,
        }
    except BaseException as exc:
        payload = failure(exc)
    out.write(RESULT_PREFIX + json.dumps(payload) + "\n")
    out.flush()
    return 0


def run_as_process(argv: Sequence[str]) -> int:
    """``main`` for a process of its own: stdout carries the result line and nothing else.

    The result line goes to a copy of file descriptor 1, and descriptor 1 itself is pointed at
    stderr. Whatever a library prints, from Python or straight from C, cannot spoil the line.
    """
    result = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    return main(argv, result)


# --- Runs ---


def _run_once(files: ExampleFiles, engine: str) -> list[Output]:
    # ``run`` takes lists and dicts of several kinds of items. mypy takes a ``list[Path]`` for
    # none of those, since a list is invariant, so these are copies that mypy does not check.
    structures: list[Any] = [*files.structures]
    datapoints: dict[str, Any] = {**files.datapoints}
    value_domains: list[Any] = [*files.value_domains]
    external_routines: list[Any] = [*files.external_routines]
    results = run(
        script=files.script,
        data_structures=structures,
        datapoints=datapoints,
        value_domains=value_domains or None,
        external_routines=external_routines or None,
        return_only_persistent=False,
        use_duckdb=engine == "duckdb",
    )
    return serialise(results)


def _dataset(name: str, dataset: Dataset) -> ActualDataset:
    components = {
        component_name: ActualComponent(
            normalize_role(component.role.value),
            _type_name(component.data_type),
            component.nullable,
        )
        for component_name, component in dataset.components.items()
    }
    rows: list[dict[str, str | None]] = []
    if dataset.data is not None:
        columns = [dataset.data[component_name].tolist() for component_name in components]
        rows = [dict(zip(components, map(_text, cells))) for cells in zip(*columns)]
    return ActualDataset(name, components, rows)


def _type_name(data_type: Any) -> str:
    """The canonical VTL name of a type, which vtlengine holds as a class, or as an instance."""
    kind = data_type if inspect.isclass(data_type) else type(data_type)
    return normalize_type(kind.__name__)


def _text(value: Any) -> str | None:
    return None if pd.isna(value) else str(value)


# --- Command line ---


def _repeat(text: str) -> int:
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a number of runs of at least 1")
    return number


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """``<example folder> <engine> <repeat>``; a usage error exits with 2."""
    parser = argparse.ArgumentParser(
        prog="python -m tool.child",
        description="Run one example on one engine and print RESULT:<json>.",
    )
    parser.add_argument("example", type=Path, help="the example folder")
    parser.add_argument("engine", choices=ENGINES)
    parser.add_argument("repeat", type=_repeat, help="how many times to run the script")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run_as_process(sys.argv[1:]))
