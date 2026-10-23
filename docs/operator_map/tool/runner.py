"""Parent side of the runner: an example on an engine, in a child process that can be killed.

A run happens in ``python -m tool.child`` (see ``tool.child``), so that a run that hangs, takes
all the memory or crashes the interpreter costs one example and not the generator. The child
starts a session of its own, so that on a timeout it is killed together with everything it
started. Nothing here raises for what happens to a child: every outcome is a ``RunOutput``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from tool.compare import ActualComponent, ActualDataset, ActualScalar

Engine = Literal["pandas", "duckdb"]

_RESULT_PREFIX: Final = "RESULT:"  # the line of the child that carries its result
_MAX_MESSAGE: Final = 300
_ATTEMPTS: Final = 2  # a run that times out is tried once more
_WORKTREE: Final = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RunOutput:
    """What an example gave on an engine: its outputs, or the reason why there are none."""

    ok: bool
    datasets: list[ActualDataset]
    scalars: list[ActualScalar]
    error_type: str | None = None  # the class of the error; ``Timeout`` or ``Crash`` for the child
    error_code: str | None = None  # the code of a vtlengine error, such as ``2-3-6``
    error_message: str | None = None  # a single line of at most 300 characters
    stable: bool | None = None  # whether every run gave the same outputs; None for a single run


def child_env() -> dict[str, str]:
    """The environment of a child: the engine of this worktree, one thread, 1 GB for DuckDB.

    The rest is the environment of this process.
    """
    paths = [_WORKTREE / "src", _WORKTREE / "docs" / "operator_map"]
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(str(path) for path in paths),
        "VTL_THREADS": "1",
        "VTL_MEMORY_LIMIT": "1GB",
    }


def run_example(path: Path, engine: Engine, timeout_s: float, repeat: int = 1) -> RunOutput:
    """Run the example folder ``path`` on ``engine`` in a child process.

    ``repeat`` is how many times the child runs the script; with 2 or more, ``stable`` says
    whether the runs agree. Each attempt has ``timeout_s`` seconds. When one runs out, the child
    and everything it started are killed and the run is tried once more, since a busy machine
    can make one run slow; a second timeout is the result, ``error_type`` ``Timeout``. A child
    that dies without a result line is a ``Crash``, which is not tried again.
    """
    if repeat < 1:
        raise ValueError(f"repeat must be at least 1, not {repeat}")
    for _ in range(_ATTEMPTS):
        output = _attempt(path, engine, timeout_s, repeat)
        if output is not None:
            return output
    message = f"no result in {timeout_s:g} s, {_ATTEMPTS} attempts"
    return RunOutput(False, [], [], "Timeout", None, message, None)


def run_all(
    jobs: Sequence[tuple[Path, Engine, int]], workers: int, timeout_s: float
) -> dict[tuple[Path, Engine], RunOutput]:
    """Run every ``(example folder, engine, repeat)`` job, ``workers`` at a time.

    The result has the jobs in their order, keyed by ``(example folder, engine)``. An exception,
    such as the interrupt of the caller, cancels the jobs that have not started.
    """
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [
            ((path, engine), pool.submit(run_example, path, engine, timeout_s, repeat))
            for path, engine, repeat in jobs
        ]
        return {key: future.result() for key, future in futures}
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# --- One child ---


def _child_command(path: Path, engine: Engine, repeat: int) -> list[str]:
    return [sys.executable, "-m", "tool.child", str(path), engine, str(repeat)]


def _attempt(path: Path, engine: Engine, timeout_s: float, repeat: int) -> RunOutput | None:
    """One child process: its output, or None when it ran out of time and was killed."""
    try:
        process = subprocess.Popen(  # noqa: S603
            _child_command(path, engine, repeat),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env(),
            start_new_session=True,
        )
    except OSError as exc:
        return _crash(f"cannot start the child: {exc}")
    with process:
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            return None
        except BaseException:
            # The caller was interrupted. The child is in a session of its own, where an
            # interrupt of the terminal does not reach it: it must not be left running.
            _kill_group(process)
            raise
    return _decode(stdout, stderr, process.returncode)


def _kill_group(process: subprocess.Popen[Any]) -> None:
    """Kill the child and everything it started: the child leads a process group of its own."""
    with suppress(ProcessLookupError):  # nothing is left in the group
        os.killpg(process.pid, signal.SIGKILL)


def _decode(stdout: str, stderr: str, returncode: int) -> RunOutput:
    """The output of a child that ended: the last result line it printed, else a crash."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                return _output(json.loads(line[len(_RESULT_PREFIX) :]))
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                return _crash(f"exit {returncode}: unreadable result ({type(exc).__name__}: {exc})")
    last = next((text for text in map(str.strip, reversed(stderr.splitlines())) if text), "")
    return _crash(f"exit {returncode}: {last}" if last else f"exit {returncode}")


def _crash(message: str) -> RunOutput:
    return RunOutput(False, [], [], "Crash", None, message[:_MAX_MESSAGE], None)


def _output(payload: Any) -> RunOutput:
    """The ``RunOutput`` of the JSON of a result line; the shapes are in ``tool.child``."""
    return RunOutput(
        ok=bool(payload["ok"]),
        datasets=[_dataset(item) for item in payload.get("datasets", [])],
        scalars=[ActualScalar(**item) for item in payload.get("scalars", [])],
        error_type=payload.get("error_type"),
        error_code=payload.get("error_code"),
        error_message=payload.get("error_message"),
        stable=payload.get("stable"),
    )


def _dataset(item: Any) -> ActualDataset:
    components = {name: ActualComponent(**fields) for name, fields in item["components"].items()}
    return ActualDataset(item["name"], components, item["rows"])
