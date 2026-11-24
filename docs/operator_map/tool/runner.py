"""Parent side of the runner: an example on an engine, in a child process that can be killed.

A run happens in ``python -m tool.child`` (see ``tool.child``), so that a run that hangs, takes
all the memory or crashes the interpreter costs one example and not the generator. The child
starts a session of its own, so that on a timeout it is killed together with everything it
started. Nothing here raises for what happens to a child: every outcome is a ``RunOutput``.

An interrupt of the terminal reaches neither the children, in sessions of their own, nor the
threads of ``run_all`` that wait on them: only the main thread gets it. So ``run_all`` keeps the
children that run, and an exception in the main thread, an interrupt included, cancels the run:
the children that run are killed, and no child starts after that.

``preflight`` starts one child before a run, to see that it can import the engines at all: a
child that cannot would make every example a crash.

A child is forked from a server that imported vtlengine, pandas and DuckDB once (the
``forkserver`` of ``multiprocessing``), where the platform can fork: it starts with the engines
in memory, so that a run takes a tenth of the time of a fresh interpreter, most of which goes to
the imports. It is still a process of its own for every run, with a session of its own, its
output in files and its exit code, and the server never runs an example, so a run cannot see
what another one did. ``run_all(fresh=True)``, and a platform that cannot fork, start a fresh
interpreter for every child instead (``python -m tool.child``).
"""

from __future__ import annotations

import functools
import json
import multiprocessing
import multiprocessing.forkserver
import os
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, Final, Literal

from tool.compare import ActualComponent, ActualDataset, ActualScalar

Engine = Literal["pandas", "duckdb"]

_RESULT_PREFIX: Final = "RESULT:"  # the line of the child that carries its result
_MAX_MESSAGE: Final = 300
_ATTEMPTS: Final = 2  # a run that times out is tried once more
_WORKTREE: Final = Path(__file__).resolve().parents[3]
_PREFLIGHT_PREFIX: Final = "PREFLIGHT:"  # the line of the preflight child that carries its answer
# The preflight child imports what every run needs: vtlengine with its compiled parser, both
# engines, and the module of the child. It answers with the versions of the engines.
_PREFLIGHT: Final = """\
import json

import duckdb
import pandas

import tool.child
import vtlengine
import vtlengine.duckdb_transpiler
import vtlengine.Interpreter

vtlengine.check_parser()
print("PREFLIGHT:" + json.dumps({"pandas": pandas.__version__, "duckdb": duckdb.__version__}))
"""
_PREFLIGHT_HINT: Final = (
    "run the generator with a Python that can import vtlengine: the compiled parser must be in src/"
)
# What every child has in its environment besides that of this process: one thread and 1 GB
# for DuckDB.
_CHILD_SETTINGS: Final = {"VTL_THREADS": "1", "VTL_MEMORY_LIMIT": "1GB"}
# Where a child imports from: the engine of this worktree, and the modules of the map.
_CHILD_PATHS: Final = (_WORKTREE / "src", _WORKTREE / "docs" / "operator_map")
# What the server of the forked children imports once, as the preflight child does.
_PRELOAD: Final = (
    "duckdb",
    "pandas",
    "vtlengine",
    "vtlengine.duckdb_transpiler",
    "vtlengine.Interpreter",
    "tool.child",
)


@dataclass(frozen=True)
class RunOutput:
    """What an example gave on an engine: its outputs, or the reason why there are none."""

    ok: bool
    datasets: list[ActualDataset]
    scalars: list[ActualScalar]
    # The class of the error: ``Timeout`` or ``Crash`` for the child, ``Cancelled`` for a job of
    # a run that was cancelled (see ``run_all``).
    error_type: str | None = None
    error_code: str | None = None  # the code of a vtlengine error, such as ``2-3-6``
    error_message: str | None = None  # a single line of at most 300 characters
    stable: bool | None = None  # whether every run gave the same outputs; None for a single run


@dataclass(frozen=True)
class EngineVersions:
    """The versions of the engines that the children import."""

    pandas: str
    duckdb: str


class PreflightError(Exception):
    """A child cannot import vtlengine and its engines: the reason, in one line."""


def child_env() -> dict[str, str]:
    """The environment of a child: the engine of this worktree, one thread, 1 GB for DuckDB.

    The rest is the environment of this process.
    """
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(str(path) for path in _CHILD_PATHS),
        **_CHILD_SETTINGS,
    }


def forking_available() -> bool:
    """Whether the children can be forked from a server: not on Windows."""
    return "forkserver" in multiprocessing.get_all_start_methods()


def _main_can_run_again() -> bool:
    """Whether a forked child can run the main module of this process again, as it does.

    ``multiprocessing`` runs it again in every child, by its module name or by its path: a
    script on disk or a module can be, while ``python -c`` and ``python -`` have no file.
    """
    main = sys.modules.get("__main__")
    if getattr(main, "__spec__", None) is not None:
        return True
    path = getattr(main, "__file__", None)
    return path is not None and Path(path).is_file()


def _main_preload() -> list[str]:
    """The script of this process by its module name, when the server can import it so.

    Every child runs the script again: with its imports in the server, that takes a tenth of
    the time (the generator's catalogue and pages, for ``generate.py``).
    """
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if getattr(main, "__spec__", None) is not None or path is None:
        return []
    script = Path(path).resolve()
    folders = {str(Path(entry).resolve()) for entry in sys.path if entry}
    return [script.stem] if script.is_file() and str(script.parent) in folders else []


@functools.cache
def _forkserver() -> BaseContext:
    """The context of the forked children, whose server imports the engines of this worktree.

    The server starts with the paths of a child first on its ``PYTHONPATH``: Python 3.12 and
    older do not give it the ``sys.path`` of this process, and it would import the vtlengine
    that its own path finds first, an installed one or another checkout's. Python 3.13 gives
    it that ``sys.path``, so the paths go first there too. This process still imports none of
    the engines.
    """
    for path in reversed(_CHILD_PATHS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload([*_PRELOAD, *_main_preload()])
    before = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in _CHILD_PATHS), *([before] if before else [])]
    )
    try:
        multiprocessing.forkserver.ensure_running()
    finally:
        if before is None:
            del os.environ["PYTHONPATH"]
        else:
            os.environ["PYTHONPATH"] = before
    return context


def _engine_problem() -> str | None:
    """Why the vtlengine of a forked child is not the one of this worktree; None if it is.

    The child has it from the server, which imported it before any child: one from elsewhere
    would run every example on another engine, so a child says so instead of running.
    """
    module = sys.modules.get("vtlengine")
    path = getattr(module, "__file__", None)
    if path is None:
        return None
    expected = (_WORKTREE / "src").resolve()
    if Path(path).resolve().is_relative_to(expected):
        return None
    return (
        f"the forked runs import vtlengine from {path}, not from {expected}: run the "
        "generator with --fresh-processes"
    )


def _forked_main(command: Sequence[str], out_path: str, err_path: str) -> None:
    """In a child forked from the server: ``command`` as a fresh interpreter would run it.

    ``command`` is ``python -m <module> <arguments>`` or ``python -c <code> <arguments>``. The
    child leads a session of its own, as a fresh child does, and its standard output and error
    go to the files that the parent reads when it ends.
    """
    os.setsid()
    for fd, path in ((1, out_path), (2, err_path)):
        target = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(target, fd)
        os.close(target)
    os.environ.update(_CHILD_SETTINGS)
    problem = _engine_problem()
    if problem is not None:
        raise RuntimeError(problem)
    flag, what, *arguments = command[1:]
    if flag == "-m":
        sys.argv = [what, *arguments]
        runpy.run_module(what, run_name="__main__", alter_sys=True)
    else:
        sys.argv = ["-c", *arguments]
        exec(compile(what, "<string>", "exec"), {"__name__": "__main__"})  # noqa: S102


def _forkable(command: Sequence[str]) -> bool:
    """Whether ``command`` is ``python -m`` or ``python -c``, which a forked child can run."""
    return len(command) >= 3 and command[0] == sys.executable and command[1] in ("-m", "-c")


class _Forked:
    """A child forked from the server, with what the runner uses of ``subprocess.Popen``."""

    def __init__(self, context: BaseContext, command: Sequence[str]) -> None:
        self._folder = Path(tempfile.mkdtemp(prefix="vtl-map-run-"))
        self._out, self._err = self._folder / "out", self._folder / "err"
        self._process = context.Process(  # type: ignore[attr-defined]
            target=_forked_main, args=(list(command), str(self._out), str(self._err))
        )
        try:
            self._process.start()
        except BaseException:
            shutil.rmtree(self._folder, ignore_errors=True)
            raise
        self.pid: int = self._process.pid

    @property
    def returncode(self) -> int | None:
        """The exit code once the child has ended, negative for a signal; None until then."""
        code: int | None = self._process.exitcode
        return code

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        """Wait for the end of the child, then its output and its error, as text."""
        self.wait(timeout)
        return self._read(self._out), self._read(self._err)

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the end of the child: its exit code."""
        self._process.join(timeout)
        code = self.returncode
        if code is None:
            raise subprocess.TimeoutExpired("forked child", timeout or 0.0)
        return code

    def kill_group(self) -> None:
        """Kill the child and what it started; the child alone if it has no group yet.

        A child that has not called ``setsid`` yet, just after its fork, leads no group. One
        that has ended is left alone: the server reaps it at once, and its id may have been
        given to another process.
        """
        if self.returncode is not None:
            return
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            with suppress(ProcessLookupError):
                os.kill(self.pid, signal.SIGKILL)

    def __enter__(self) -> _Forked:
        return self

    def __exit__(self, *_: object) -> None:
        self._process.join()
        shutil.rmtree(self._folder, ignore_errors=True)

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:  # a child killed before it opened its files
            return ""


class _Children:
    """The children of a run that are running, and whether the run is cancelled.

    A child is started under the lock and kept at once, and ``cancel`` kills the children that
    are kept under the same lock: a child is either killed by the cancel or never started.
    """

    def __init__(self, fresh: bool = False) -> None:
        self._lock = threading.Lock()
        self._live: set[subprocess.Popen[str] | _Forked] = set()
        self._cancelled = threading.Event()
        forked = not fresh and forking_available() and _main_can_run_again()
        self._context = _forkserver() if forked else None

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def start(self, command: list[str]) -> subprocess.Popen[str] | _Forked | None:
        """Start a child and keep it; None, and nothing started, once the run is cancelled.

        The child is forked from the server when there is one and ``command`` is a Python
        module or code, else it is a fresh process.
        """
        with self._lock:
            if self._cancelled.is_set():
                return None
            if self._context is not None and _forkable(command):
                forked = _Forked(self._context, command)
                self._live.add(forked)
                return forked
            process = subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=child_env(),
                start_new_session=True,
            )
            self._live.add(process)
            return process

    def forget(self, process: subprocess.Popen[str] | _Forked) -> None:
        """Stop keeping a child that has ended."""
        with self._lock:
            self._live.discard(process)

    def cancel(self) -> None:
        """Cancel the run: no child starts from now on, and the children that run are killed."""
        with self._lock:
            self._cancelled.set()
            for process in self._live:
                # A child that was waited for has given its id back, which may be reused.
                if process.returncode is None:
                    _kill_group(process)


def run_example(
    path: Path,
    engine: Engine,
    timeout_s: float,
    repeat: int = 1,
    *,
    children: _Children | None = None,
) -> RunOutput:
    """Run the example folder ``path`` on ``engine`` in a child process.

    ``repeat`` is how many times the child runs the script; with 2 or more, ``stable`` says
    whether the runs agree. Each attempt has ``timeout_s`` seconds. When one runs out, the child
    and everything it started are killed and the run is tried once more, since a busy machine
    can make one run slow; a second timeout is the result, ``error_type`` ``Timeout``. A child
    that dies without a result line is a ``Crash``, which is not tried again.

    ``children`` are those of the ``run_all`` that runs the example. Once that run is cancelled,
    no child starts and a run that timed out is not tried again: the result is ``Cancelled``.
    """
    if repeat < 1:
        raise ValueError(f"repeat must be at least 1, not {repeat}")
    children = _Children() if children is None else children
    for _ in range(_ATTEMPTS):
        output = _attempt(path, engine, timeout_s, repeat, children)
        if output is not None:
            return output
        if children.cancelled:
            return _cancelled()
    message = f"no result in {timeout_s:g} s, {_ATTEMPTS} attempts"
    return RunOutput(False, [], [], "Timeout", None, message, None)


def run_all(
    jobs: Sequence[tuple[Path, Engine, int]],
    workers: int,
    timeout_s: float,
    progress: Callable[[int, int, tuple[Path, Engine]], None] | None = None,
    *,
    fresh: bool = False,
) -> dict[tuple[Path, Engine], RunOutput]:
    """Run every ``(example folder, engine, repeat)`` job, ``workers`` at a time.

    ``fresh`` starts a fresh interpreter for each child, instead of forking it from the server.

    ``progress`` is called, in this thread, each time a job ends: with how many have ended, how
    many there are, and the key of the one that ended.

    The result has the jobs in their order, keyed by ``(example folder, engine)``. An exception
    in the thread that waits for the results, such as the interrupt of the caller or an error
    that a job raised, cancels the run before it goes on: the children that run are killed with
    everything they started, and no child starts after that, neither for a job that waits nor
    for a second attempt of a run that timed out.
    """
    children = _Children(fresh)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [
            (
                (path, engine),
                pool.submit(run_example, path, engine, timeout_s, repeat, children=children),
            )
            for path, engine, repeat in jobs
        ]
        keys = {future: key for key, future in futures}
        for done, future in enumerate(as_completed(keys), start=1):
            future.result()  # a job that raised stops the run as soon as it ends
            if progress is not None:
                progress(done, len(futures), keys[future])
        return {key: future.result() for key, future in futures}
    except BaseException:
        # Only this thread gets an interrupt: the workers wait on children that are in sessions
        # of their own, which the interrupt of the terminal does not reach. So the run is
        # cancelled here, and the pool is not left to wait for the children to time out.
        children.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# --- The preflight ---


def preflight(timeout_s: float) -> EngineVersions:
    """Check that a child can import vtlengine and both engines, before any example runs.

    The child imports what every run needs: vtlengine with its compiled parser, the pandas and
    the DuckDB engines, and ``tool.child``. It answers with the versions of pandas and DuckDB.
    Anything else, a child that fails, hangs past ``timeout_s`` or gives no readable answer,
    raises ``PreflightError`` with the reason in one line.
    """
    try:
        done = subprocess.run(  # noqa: S603
            _preflight_command(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env(),
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise PreflightError(_preflight_line(f"no answer in {timeout_s:g} s")) from None
    except OSError as exc:
        raise PreflightError(_preflight_line(f"it cannot start: {exc}")) from exc
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(_PREFLIGHT_PREFIX):
            try:
                versions = json.loads(line[len(_PREFLIGHT_PREFIX) :])
                return EngineVersions(str(versions["pandas"]), str(versions["duckdb"]))
            except (ValueError, KeyError, TypeError) as exc:
                reason = f"an unreadable answer, {type(exc).__name__}"
                raise PreflightError(_preflight_line(reason)) from exc
    last = next((text for text in map(str.strip, reversed(done.stderr.splitlines())) if text), "")
    raise PreflightError(_preflight_line(last or f"exit {done.returncode} and no answer"))


def _preflight_command() -> list[str]:
    return [sys.executable, "-c", _PREFLIGHT]


def _preflight_line(reason: str) -> str:
    """The line of a preflight that failed: what happened, and what the generator needs."""
    return (
        "preflight: a child process cannot import vtlengine and its engines "
        f"({reason[:_MAX_MESSAGE]}); {_PREFLIGHT_HINT}"
    )


# --- One child ---


def _child_command(path: Path, engine: Engine, repeat: int) -> list[str]:
    return [sys.executable, "-m", "tool.child", str(path), engine, str(repeat)]


def _attempt(
    path: Path, engine: Engine, timeout_s: float, repeat: int, children: _Children
) -> RunOutput | None:
    """One child process: its output, or None when it ran out of time and was killed."""
    try:
        process = children.start(_child_command(path, engine, repeat))
    except OSError as exc:
        return _crash(f"cannot start the child: {exc}")
    if process is None:
        return _cancelled()
    try:
        with process:
            try:
                stdout, stderr = process.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill_group(process)
                return None
            except BaseException:
                # Only the main thread gets an interrupt, so this is ``run_example`` called in
                # the main thread: under ``run_all`` the workers never see one, and ``run_all``
                # kills its children itself. The child is in a session of its own, where an
                # interrupt of the terminal does not reach it: it must not be left running.
                _kill_group(process)
                raise
    finally:
        children.forget(process)
    return _decode(stdout, stderr, process.wait())  # it has ended: wait gives its exit code


def _kill_group(process: subprocess.Popen[Any] | _Forked) -> None:
    """Kill the child and everything it started: the child leads a process group of its own."""
    if isinstance(process, _Forked):
        process.kill_group()
        return
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


def _cancelled() -> RunOutput:
    """The output of a job of a run that was cancelled before its child could give one."""
    return RunOutput(False, [], [], "Cancelled", None, "the run was cancelled", None)


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
