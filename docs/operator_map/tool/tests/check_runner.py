# ruff: noqa: S101
"""Checks for the runner: one example on one engine, in a child process of its own.

The checks that run an engine are real runs of the engines of this worktree in a child process,
on a tiny example, so each takes about a second. The other checks put a few lines of Python in
place of the child process (``fake_child``), to pin how the runner treats a child that crashes,
hangs, babbles or answers on its second attempt.
"""

from __future__ import annotations

import io
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from textwrap import dedent
from typing import Any

import duckdb
import pandas as pd
import pytest

from tool import child, runner
from tool.compare import ActualComponent, ActualDataset, ActualScalar, Output, compare_dataset
from tool.examples import ERROR_CODE, load_example
from tool.runner import Engine, RunOutput, child_env, run_all, run_example
from vtlengine.DataTypes import Boolean, Integer, Null, Number, String, TimeInterval, TimePeriod
from vtlengine.Exceptions import VTLEngineException
from vtlengine.Model import Component, Dataset, Role, Scalar

TIMEOUT = 120.0  # the tiny examples take about a second: the limit only has to be finite
ENGINES: tuple[Engine, ...] = ("pandas", "duckdb")
WORKTREE = Path(__file__).resolve().parents[4]
CODE = re.compile(r"^\d+(-\d+)+$")

# The first Addition example of the manual: DS_r := DS_1 + DS_2, added where the identifiers meet.
ADDITION = "DS_r := DS_1 + DS_2;\n"
HEADER = "Id_1,Id_2,Me_1,Me_2\n"
DS_1_ROWS = "10,A,5,5.0\n10,B,2,10.5\n11,A,3,12.2\n11,B,4,20.3\n"
DS_2_ROWS = "10,A,10,3.0\n10,C,8,4.0\n11,B,6,7.0\n12,A,12,7.0\n"
RESULT_ROWS = "10,A,15,8.0\n11,B,10,27.3\n"
COMPONENTS = [
    ("Id_1", "Identifier", "Integer", False),
    ("Id_2", "Identifier", "String", False),
    ("Me_1", "Measure", "Integer", True),
    ("Me_2", "Measure", "Number", True),
]

# A value of every kind that the examples return, and a data point of nulls.
TYPES_COMPONENTS = [
    ("Id_1", "Identifier", "Integer", False),
    ("Me_1", "Measure", "Integer", True),
    ("Me_2", "Measure", "Number", True),
    ("Me_3", "Measure", "Boolean", True),
    ("Me_4", "Measure", "Date", True),
    ("Me_5", "Measure", "Time_Period", True),
    ("Me_6", "Measure", "String", True),
    ("At_1", "Attribute", "String", True),
]
TYPES_HEADER = "Id_1,Me_1,Me_2,Me_3,Me_4,Me_5,Me_6,At_1\n"
TYPES_ROWS = '1,5,1.5,true,2020-01-31,2020Q1,"a,b",x\n2,,,,,,,\n'
TYPES_SCRIPT = "DS_r := DS_1;\nsc_i := 3 + 5;\nsc_b := true;\n"

# An example that needs the two extra inputs of ``run``: a value domain and an external routine.
EXTRAS_SCRIPT = (
    "DS_r := DS_1 [calc Me_2 := Me_1 in Countries];\n"
    'DS_s := eval(SQL_1(DS_1) language "SQL" returns dataset '
    "{ identifier<integer> Id_1, measure<string> Me_1 });\n"
)
EXTRAS_COMPONENTS = [("Id_1", "Identifier", "Integer", False), ("Me_1", "Measure", "String", True)]
VALUE_DOMAIN = {"name": "Countries", "type": "String", "setlist": ["DE", "FR", "IT"]}
EXTERNAL_ROUTINE = {"name": "SQL_1", "query": "SELECT Id_1, Me_1 FROM DS_1 WHERE Id_1 = 1;"}

# What a child prints when it answers, and what the runner makes of it.
OK_PAYLOAD: dict[str, Any] = {
    "ok": True,
    "datasets": [
        {
            "name": "DS_r",
            "components": {
                "Id_1": {"role": "Identifier", "type": "Integer", "nullable": False},
                "Me_1": {"role": "Measure", "type": "Number", "nullable": True},
            },
            "rows": [{"Id_1": "1", "Me_1": "2.5"}, {"Id_1": "2", "Me_1": None}],
        }
    ],
    "scalars": [{"name": "sc_r", "type": "Integer", "value": "8"}],
    "stable": True,
}
OK_OUTPUT = RunOutput(
    ok=True,
    datasets=[
        ActualDataset(
            "DS_r",
            {
                "Id_1": ActualComponent("Identifier", "Integer", False),
                "Me_1": ActualComponent("Measure", "Number", True),
            },
            [{"Id_1": "1", "Me_1": "2.5"}, {"Id_1": "2", "Me_1": None}],
        )
    ],
    scalars=[ActualScalar("sc_r", "Integer", "8")],
    stable=True,
)

# Children that are not the real one: Python code that gets ``<example> <engine> <repeat>``.
HANG = "import time\ntime.sleep(60)\n"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def structure(name: str, components: list[tuple[str, str, str, bool]]) -> str:
    """The text of a structure file that describes one dataset."""
    items = [
        {"name": component, "role": role, "type": type_, "nullable": nullable}
        for component, role, type_, nullable in components
    ]
    return json.dumps({"datasets": [{"name": name, "DataStructure": items}]})


def build_example(path: Path, script: str = ADDITION) -> Path:
    """DS_1 + DS_2 with the numbers of the manual, and its expected result DS_r."""
    write(path / "script.vtl", script)
    for name, rows in (("DS_1", DS_1_ROWS), ("DS_2", DS_2_ROWS)):
        write(path / f"input/{name}.json", structure(name, COMPONENTS))
        write(path / f"input/{name}.csv", HEADER + rows)
    write(path / "expected/DS_r.json", structure("DS_r", COMPONENTS))
    write(path / "expected/DS_r.csv", HEADER + RESULT_ROWS)
    return path


def build_types_example(path: Path) -> Path:
    """A dataset with a value of each kind, copied into DS_r, and two scalars."""
    write(path / "script.vtl", TYPES_SCRIPT)
    write(path / "input/DS_1.json", structure("DS_1", TYPES_COMPONENTS))
    write(path / "input/DS_1.csv", TYPES_HEADER + TYPES_ROWS)
    write(path / "expected/DS_r.json", structure("DS_r", TYPES_COMPONENTS))
    write(path / "expected/DS_r.csv", TYPES_HEADER + TYPES_ROWS)
    return path


def build_extras_example(path: Path) -> Path:
    """A calc with a value domain, and an eval of an external routine."""
    write(path / "script.vtl", EXTRAS_SCRIPT)
    write(path / "input/DS_1.json", structure("DS_1", EXTRAS_COMPONENTS))
    write(path / "input/DS_1.csv", "Id_1,Me_1\n1,DE\n2,XX\n")
    write(path / "value_domains/countries.json", json.dumps(VALUE_DOMAIN))
    # The engine names a routine after its file, not after the "name" in it.
    write(path / "external_routines/SQL_1.json", json.dumps(EXTERNAL_ROUTINE))
    write(path / "expected/DS_r.json", structure("DS_r", EXTRAS_COMPONENTS))
    write(path / "expected/DS_r.csv", "Id_1,Me_1\n1,DE\n2,XX\n")
    return path


def fake_child(monkeypatch: pytest.MonkeyPatch, *codes: str) -> list[list[str]]:
    """Run Python code instead of ``python -m tool.child``, one snippet per attempt.

    The code gets the arguments of the real child, ``<example> <engine> <repeat>``. The last
    snippet serves every attempt after it. The result lists the arguments of the children that
    were started.
    """
    started: list[list[str]] = []

    def command(path: Path, engine: Engine, repeat: int) -> list[str]:
        code = codes[min(len(started), len(codes) - 1)]
        started.append([str(path), engine, str(repeat)])
        return [sys.executable, "-c", code, str(path), engine, str(repeat)]

    monkeypatch.setattr(runner, "_child_command", command)
    return started


def printing(payload: dict[str, Any], before: str = "", after: str = "") -> str:
    """Python code that prints ``payload`` as the result line, between two other statements."""
    return f"{before}\nprint('RESULT:' + {json.dumps(payload)!r})\n{after}\n"


def process_of_its_own(code: str, *args: str) -> subprocess.Popen[str]:
    """A process that starts a session of its own, as the runner starts its children."""
    return subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", code, *args],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def run_child(*command: str) -> subprocess.CompletedProcess[str]:
    """``python <command>`` with the environment of the runner, to the end."""
    return subprocess.run(  # noqa: S603
        [sys.executable, *command],
        capture_output=True,
        text=True,
        env=child_env(),
        timeout=TIMEOUT,
    )


# --- Real engines ---


def test_runs_on_both_engines(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    expected = load_example(path).expected_datasets[0]
    for engine in ENGINES:
        out = run_example(path, engine, TIMEOUT)

        assert out.ok, (engine, out.error_type, out.error_message)
        assert [dataset.name for dataset in out.datasets] == ["DS_r"], engine
        result = out.datasets[0]
        assert result.components == {
            "Id_1": ActualComponent("Identifier", "Integer", False),
            "Id_2": ActualComponent("Identifier", "String", False),
            "Me_1": ActualComponent("Measure", "Integer", True),
            "Me_2": ActualComponent("Measure", "Number", True),
        }, engine
        assert result.components["Me_1"].type == "Integer", engine
        assert len(result.rows) == 2, engine
        assert all(isinstance(value, str) for row in result.rows for value in row.values())
        # What comes back is what the comparator judges: the manual's result.
        assert compare_dataset(expected, result) is None, engine
        assert out.scalars == [], engine
        assert (out.error_type, out.error_code, out.error_message) == (None, None, None), engine
        assert out.stable is None, engine  # a single run has no stability to tell


def test_errors_are_captured(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex", script="DS_r := DS_1 + DS_9;\n")
    for engine in ENGINES:
        out = run_example(path, engine, TIMEOUT)

        assert out.ok is False, engine
        assert out.error_type == "SemanticError", engine
        assert CODE.match(out.error_code or ""), engine
        # The error of vtlengine is a (message, code) pair: the message is its first element.
        message = out.error_message or ""
        assert message.startswith("Dataset or Scalar DS_9 not found"), (engine, message)
        assert "\n" not in message, engine
        assert len(message) <= 300, engine
        assert (out.datasets, out.scalars, out.stable) == ([], [], None), engine


def test_timeout_is_an_error_not_a_hang(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex")
    started = time.monotonic()
    out = run_example(path, "pandas", timeout_s=0.01)

    assert time.monotonic() - started < 10
    assert out.ok is False
    assert out.error_type == "Timeout"
    assert out.datasets == []
    assert out.scalars == []
    assert out.error_code is None
    assert out.stable is None


def test_random_stability_flag(tmp_path: Path) -> None:
    path = build_example(tmp_path / "ex", script="DS_r := random(DS_1, 5);\n")
    for engine in ENGINES:
        out = run_example(path, engine, TIMEOUT, repeat=2)

        assert out.ok, (engine, out.error_type, out.error_message)
        assert out.stable is True, engine


def test_every_kind_of_value_is_serialised_as_text_and_nulls_as_none(tmp_path: Path) -> None:
    path = build_types_example(tmp_path / "ex")
    for engine in ENGINES:
        out = run_example(path, engine, TIMEOUT)

        assert out.ok, (engine, out.error_type, out.error_message)
        (result,) = out.datasets
        assert result.name == "DS_r"
        assert result.components == {
            "Id_1": ActualComponent("Identifier", "Integer", False),
            "Me_1": ActualComponent("Measure", "Integer", True),
            "Me_2": ActualComponent("Measure", "Number", True),
            "Me_3": ActualComponent("Measure", "Boolean", True),
            "Me_4": ActualComponent("Measure", "Date", True),
            "Me_5": ActualComponent("Measure", "Time_Period", True),
            "Me_6": ActualComponent("Measure", "String", True),
            "At_1": ActualComponent("Attribute", "String", True),
        }, engine
        rows = {row["Id_1"]: row for row in result.rows}
        assert rows["1"] == {
            "Id_1": "1",
            "Me_1": "5",
            "Me_2": "1.5",
            "Me_3": "True",
            "Me_4": "2020-01-31",
            "Me_5": "2020Q1",
            "Me_6": "a,b",
            "At_1": "x",
        }, engine
        assert rows["2"] == {"Id_1": "2", **dict.fromkeys(list(rows["1"])[1:])}, engine
        assert out.scalars == [
            ActualScalar("sc_i", "Integer", "8"),
            ActualScalar("sc_b", "Boolean", "True"),
        ], engine


def test_value_domains_and_external_routines_reach_the_engines(tmp_path: Path) -> None:
    path = build_extras_example(tmp_path / "ex")
    for engine in ENGINES:
        out = run_example(path, engine, TIMEOUT)

        assert out.ok, (engine, out.error_type, out.error_message)
        results = {dataset.name: dataset for dataset in out.datasets}
        assert sorted(results) == ["DS_r", "DS_s"], engine
        # The value domain tells the two countries apart, and the routine keeps the first row.
        by_id = {row["Id_1"]: row for row in results["DS_r"].rows}
        assert (by_id["1"]["Me_2"], by_id["2"]["Me_2"]) == ("True", "False"), engine
        assert results["DS_s"].rows == [{"Id_1": "1", "Me_1": "DE"}], engine


def test_an_example_that_cannot_be_loaded_is_an_error_of_the_child_not_a_crash(
    tmp_path: Path,
) -> None:
    out = run_example(tmp_path / "missing", "pandas", TIMEOUT)

    assert out.ok is False
    assert out.error_type == "ExampleError"
    assert "script.vtl" in (out.error_message or "")


def test_repeat_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repeat"):
        run_example(tmp_path, "pandas", TIMEOUT, repeat=0)


# --- What the child prints ---


def test_the_child_prints_one_result_line_and_nothing_else_on_stdout(tmp_path: Path) -> None:
    # The engines are replaced by a function that writes to stdout the way a library may: from
    # Python, and straight to the file descriptor as C code does.
    driver = dedent(
        """\
        import os, sys
        from tool import child

        def noisy(files, engine):
            print('python noise')
            os.write(1, b'c noise\\n')
            return []

        child._run_once = noisy
        sys.exit(child.run_as_process(sys.argv[1:]))
        """
    )
    path = build_example(tmp_path / "ex")
    process = run_child("-c", driver, str(path), "pandas", "1")

    assert process.returncode == 0, process.stderr
    (line,) = process.stdout.splitlines()
    assert json.loads(line.removeprefix("RESULT:")) == {
        "ok": True,
        "datasets": [],
        "scalars": [],
        "stable": None,
    }
    assert line.startswith("RESULT:")
    assert "python noise" in process.stderr
    assert "c noise" in process.stderr


def test_the_child_exits_with_zero_when_the_example_fails(tmp_path: Path) -> None:
    process = run_child("-m", "tool.child", str(tmp_path / "missing"), "duckdb", "1")

    assert process.returncode == 0, process.stderr
    (line,) = process.stdout.splitlines()
    payload = json.loads(line.removeprefix("RESULT:"))
    assert payload["ok"] is False
    assert payload["error_type"] == "ExampleError"


# --- The child in this process ---


def run_main(argv: list[str]) -> tuple[int, str]:
    """``child.main`` with its result line caught: the exit code, and what was written."""
    out = io.StringIO()
    code = child.main(argv, out)
    return code, out.getvalue()


@pytest.mark.parametrize(
    "error",
    [ValueError("x"), SystemExit(3), KeyboardInterrupt(), MemoryError(), RecursionError()],
    ids=lambda error: type(error).__name__,
)
def test_main_answers_for_every_kind_of_failure(
    error: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(files: object, engine: str) -> list[Output]:
        raise error

    monkeypatch.setattr(child, "_run_once", broken)
    code, written = run_main([str(build_example(tmp_path / "ex")), "pandas", "1"])

    assert code == 0
    assert written.endswith("\n")
    (line,) = written.splitlines()
    payload = json.loads(line.removeprefix("RESULT:"))
    assert payload["ok"] is False
    assert payload["error_type"] == type(error).__name__


def test_the_result_line_is_plain_ascii_whatever_the_error_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Whatever the encoding of the pipe is, as under a C locale: no character can fail to encode.
    def broken(files: object, engine: str) -> list[Output]:
        raise ValueError("caf\u00e9 \u2013 \u65e5\u672c \ud800")

    monkeypatch.setattr(child, "_run_once", broken)
    _, written = run_main([str(build_example(tmp_path / "ex")), "pandas", "1"])

    assert written.isascii()
    payload = json.loads(written.removeprefix("RESULT:"))
    assert payload["error_message"] == "caf\u00e9 \u2013 \u65e5\u672c \ud800"


def test_main_writes_the_outputs_and_whether_the_runs_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = {"Id_1": ActualComponent("Identifier", "Integer", False)}
    dataset = ActualDataset("DS_r", components, [{"Id_1": "1"}, {"Id_1": None}])
    answers = iter(
        [
            [dataset, ActualScalar("sc_r", "Number", "0.5")],
            [dataset, ActualScalar("sc_r", "Number", "0.6")],
        ]
    )
    monkeypatch.setattr(child, "_run_once", lambda files, engine: next(answers))
    code, written = run_main([str(build_example(tmp_path / "ex")), "duckdb", "2"])

    assert code == 0
    assert json.loads(written.removeprefix("RESULT:")) == {
        "ok": True,
        "datasets": [
            {
                "name": "DS_r",
                "components": {
                    "Id_1": {"role": "Identifier", "type": "Integer", "nullable": False}
                },
                "rows": [{"Id_1": "1"}, {"Id_1": None}],
            }
        ],
        "scalars": [{"name": "sc_r", "type": "Number", "value": "0.5"}],
        "stable": False,
    }


def vtlengine_call(path: Path, engine: Engine, **more: Any) -> dict[str, Any]:
    """What the child hands to ``vtlengine.run`` for the example ``path`` of ``build_example``."""
    return {
        "script": ADDITION,
        "data_structures": [path / "input/DS_1.json", path / "input/DS_2.json"],
        "datapoints": {"DS_1": path / "input/DS_1.csv", "DS_2": path / "input/DS_2.csv"},
        "value_domains": None,
        "external_routines": None,
        # The results of every assignment, not only the ones of the persistent ones (<-).
        "return_only_persistent": False,
        "use_duckdb": engine == "duckdb",
        **more,
    }


def test_the_example_goes_to_vtlengine_as_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(child, "run", fake_run)
    plain = build_example(tmp_path / "plain")
    full = build_example(tmp_path / "full")
    write(full / "value_domains/vd.json", "{}")
    write(full / "external_routines/er.json", "{}")

    for engine in ENGINES:
        assert child._run_once(load_example(plain), engine) == []
        assert calls.pop() == vtlengine_call(plain, engine)  # no value domains: none are passed
        assert child._run_once(load_example(full), engine) == []
        assert calls.pop() == vtlengine_call(
            full,
            engine,
            value_domains=[full / "value_domains/vd.json"],
            external_routines=[full / "external_routines/er.json"],
        )
    assert calls == []


def test_the_arguments_are_the_example_the_engine_and_the_number_of_runs() -> None:
    arguments = child.parse_arguments(["some/example", "duckdb", "2"])

    assert (arguments.example, arguments.engine, arguments.repeat) == (
        Path("some/example"),
        "duckdb",
        2,
    )


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["ex"],
        ["ex", "pandas"],
        ["ex", "oracle", "1"],
        ["ex", "pandas", "0"],
        ["ex", "pandas", "n"],
    ],
)
def test_wrong_arguments_are_a_usage_error(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        child.parse_arguments(argv)

    assert stop.value.code == 2
    assert "usage: python -m tool.child" in capsys.readouterr().err


# --- Serialisation of what vtlengine returns ---


def test_serialise_datasets_and_scalars() -> None:
    components = {
        "Id_1": Component("Id_1", Integer, Role.IDENTIFIER, False),
        "Me_1": Component("Me_1", Integer, Role.MEASURE, True),
        "Me_2": Component("Me_2", Number, Role.MEASURE, True),
        "Me_3": Component("Me_3", Boolean, Role.MEASURE, True),
        "Me_4": Component("Me_4", String, Role.MEASURE, True),
        "At_1": Component("At_1", TimePeriod, Role.ATTRIBUTE, True),
        "At_2": Component("At_2", TimeInterval, Role.VIRAL_ATTRIBUTE, True),
    }
    # The columns come in another order than the components, and the nulls in every shape.
    frame = pd.DataFrame(
        {
            "At_2": ["2020-01-01/2020-12-31", None],
            "At_1": ["2020Q1", pd.NA],
            "Me_4": pd.array(["a", None], dtype="string"),
            "Me_3": pd.array([True, None], dtype="boolean"),
            "Me_2": [0.1, float("nan")],
            "Me_1": pd.array([5, None], dtype="Int64"),
            "Id_1": [1, 2],
        }
    )
    results = {
        "DS_r": Dataset("DS_r", components, frame),
        "DS_e": Dataset("DS_e", components, None),
        "sc_1": Scalar("sc_1", Integer, 8),
        "sc_2": Scalar("sc_2", Number, 1.5),
        "sc_3": Scalar("sc_3", Null, None),
        "sc_4": Scalar("sc_4", Boolean, False),
        "sc_5": Scalar("sc_5", String, ""),
        "sc_6": Scalar("sc_6", TimePeriod, "2020Q1"),
    }

    outputs = child.serialise(results)

    structure_of_ds = {
        "Id_1": ActualComponent("Identifier", "Integer", False),
        "Me_1": ActualComponent("Measure", "Integer", True),
        "Me_2": ActualComponent("Measure", "Number", True),
        "Me_3": ActualComponent("Measure", "Boolean", True),
        "Me_4": ActualComponent("Measure", "String", True),
        "At_1": ActualComponent("Attribute", "Time_Period", True),
        "At_2": ActualComponent("Viral Attribute", "Time", True),
    }
    first: dict[str, str | None] = {
        "Id_1": "1",
        "Me_1": "5",
        "Me_2": "0.1",
        "Me_3": "True",
        "Me_4": "a",
        "At_1": "2020Q1",
        "At_2": "2020-01-01/2020-12-31",
    }
    second: dict[str, str | None] = {"Id_1": "2", **dict.fromkeys(list(first)[1:])}
    assert outputs == [
        ActualDataset("DS_r", structure_of_ds, [first, second]),
        ActualDataset("DS_e", structure_of_ds, []),  # no data at all is no data point
        ActualScalar("sc_1", "Integer", "8"),
        ActualScalar("sc_2", "Number", "1.5"),
        ActualScalar("sc_3", "Null", None),
        ActualScalar("sc_4", "Boolean", "False"),
        ActualScalar("sc_5", "String", ""),  # an empty text is not a null
        ActualScalar("sc_6", "Time_Period", "2020Q1"),
    ]
    # The order of the components, not the one of the columns, is the one of the output.
    dataset = outputs[0]
    assert isinstance(dataset, ActualDataset)
    assert list(dataset.components) == list(components)
    assert all(list(row) == list(components) for row in dataset.rows)


def test_serialise_refuses_what_is_neither_a_dataset_nor_a_scalar() -> None:
    with pytest.raises(TypeError, match="DS_r"):
        child.serialise({"DS_r": 42})


def test_stability_compares_every_repetition() -> None:
    same: list[Output] = [ActualScalar("sc", "Number", "0.5")]
    other: list[Output] = [ActualScalar("sc", "Number", "0.6")]

    assert child.collect(lambda: same, 1) == (same, None)  # nothing to compare
    assert child.collect(lambda: same, 2) == (same, True)
    answers = iter([same, other])
    assert child.collect(lambda: next(answers), 2) == (same, False)
    answers = iter([same, same, other])
    assert child.collect(lambda: next(answers), 3) == (same, False)  # the last one differs


def test_error_message_is_the_first_line_of_the_text_of_the_error() -> None:
    # vtlengine raises (message, code) pairs: the text is the first element.
    error = VTLEngineException("Dataset DS_9 not found\nsecond line", code="2-3-6")
    assert error.args == ("Dataset DS_9 not found\nsecond line", "2-3-6")
    assert child.failure(error) == {
        "ok": False,
        "error_type": "VTLEngineException",
        "error_code": "2-3-6",
        "error_message": "Dataset DS_9 not found",
    }

    def message(exc: BaseException) -> str:
        text: str = child.failure(exc)["error_message"]
        return text

    assert message(ValueError("only text")) == "only text"
    assert message(KeyError("name")) == "name"  # not the repr that str() gives, "'name'"
    assert message(ValueError(5, "x")) == "(5, 'x')"  # no text first: the text of the error
    assert message(OSError(2, "No such file")) == "[Errno 2] No such file"
    assert message(ValueError()) == ""
    assert message(ValueError("\n\n  after blank lines  \nnext")) == "after blank lines"
    assert message(ValueError("a" * 400)) == "a" * 300
    assert message(ValueError("line\r\nnext")) == "line"
    # Any break of a line ends it, so that a message never spreads over several rows of a table.
    assert message(ValueError("first\rsecond")) == "first"
    assert message(ValueError("first\u2028second")) == "first"


def test_error_code_is_the_first_argument_that_looks_like_one() -> None:
    def code(*args: object) -> str | None:
        found: str | None = child.failure(Exception(*args))["error_code"]
        return found

    assert code("text", "1-2-3") == "1-2-3"
    assert code("text", "0-1-2", "3-4") == "0-1-2"
    assert code("text", 5, "10-20") == "10-20"
    assert code("2-3-6") == "2-3-6"  # the message of an error can be its code
    assert code("text") is None
    assert code("text", "12") is None  # a dash is needed
    assert code("text", "1-2-") is None
    assert code("text", "1-2-3x") is None
    assert code("text", "1-2-3\n") is None  # the whole argument has to match
    assert code("text", "a-1") is None
    assert code("text", ("1-2-3",)) is None  # a code is a text, not a tuple that holds one
    assert code() is None


@pytest.mark.parametrize(
    "text",
    [
        "2-1-15-6",
        "1-2-23",
        "0-1",
        "x",
        "",
        "2",
        "2-",
        "-2-1",
        "2-1-a",
        "2-1-15-6\n",
        "\uff12-\uff11",
    ],
)
def test_the_child_and_the_loader_take_the_same_codes(text: str) -> None:
    # The child reads the code of an exception, and the loader checks the code of an
    # ``expected/error.json``: both must call the same texts a code, so the code is written once.
    from_the_child = child.failure(Exception("message", text))["error_code"] == text
    assert from_the_child is (ERROR_CODE.fullmatch(text) is not None)


def test_error_type_is_the_class_name() -> None:
    assert child.failure(ValueError("x"))["error_type"] == "ValueError"
    assert child.failure(RecursionError("x"))["error_type"] == "RecursionError"


# --- The child process as the runner sees it ---


def test_a_child_without_a_result_line_is_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crash = "import sys\nsys.stderr.write('first\\nlast line\\n\\n')\nsys.exit(3)\n"
    fake_child(monkeypatch, crash)
    out = run_example(tmp_path, "pandas", TIMEOUT)

    assert out == RunOutput(False, [], [], "Crash", None, "exit 3: last line", None)


def test_a_child_killed_by_a_signal_is_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What the kernel does to a run that takes all the memory: no result and no last words.
    fake_child(monkeypatch, f"import os\nos.kill(os.getpid(), {int(signal.SIGKILL)})\n")
    out = run_example(tmp_path, "duckdb", TIMEOUT)

    assert out.ok is False
    assert out.error_type == "Crash"
    assert out.error_message == "exit -9"


def test_a_child_that_prints_no_result_line_is_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_child(monkeypatch, "print('not a result')\n")
    out = run_example(tmp_path, "pandas", TIMEOUT)

    assert (out.ok, out.error_type, out.error_message) == (False, "Crash", "exit 0")


def test_output_that_is_not_utf8_is_read_all_the_same(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = "import sys\nsys.stderr.buffer.write(b'\\xff\\xfe broken\\n')\nsys.exit(3)\n"
    fake_child(monkeypatch, code)
    out = run_example(tmp_path, "pandas", TIMEOUT)

    assert out.error_type == "Crash"
    assert out.error_message == "exit 3: \ufffd\ufffd broken"


def test_the_message_of_a_crash_is_clipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_child(monkeypatch, "import sys\nsys.stderr.write('x' * 1000)\nsys.exit(1)\n")
    out = run_example(tmp_path, "pandas", TIMEOUT)

    assert out.error_type == "Crash"
    assert out.error_message == "exit 1: " + "x" * 292  # 300 characters in all


def test_the_result_line_is_found_among_other_output_and_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A child that babbles on stdout and exits badly once it answered: the answer counts.
    code = printing(
        OK_PAYLOAD, before="print('noise')", after="print('more noise')\nimport sys\nsys.exit(7)"
    )
    fake_child(monkeypatch, code)

    assert run_example(tmp_path, "pandas", TIMEOUT) == OK_OUTPUT


def test_the_last_result_line_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stale = {"ok": False, "error_type": "Stale", "error_code": None, "error_message": "old"}
    fake_child(
        monkeypatch, printing(OK_PAYLOAD, before=f"print('RESULT:' + {json.dumps(stale)!r})")
    )

    assert run_example(tmp_path, "pandas", TIMEOUT) == OK_OUTPUT


def test_an_error_result_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "ok": False,
        "error_type": "SemanticError",
        "error_code": "1-2-3",
        "error_message": "boom",
    }
    fake_child(monkeypatch, printing(payload))

    assert run_example(tmp_path, "duckdb", TIMEOUT) == RunOutput(
        False, [], [], "SemanticError", "1-2-3", "boom", None
    )


@pytest.mark.parametrize(
    "line",
    [
        "RESULT:{not json",
        "RESULT:[1, 2]",
        'RESULT:{"datasets": []}',
        'RESULT:{"ok": true, "datasets": [{"name": "DS_r"}], "scalars": []}',
        'RESULT:{"ok": true, "datasets": "DS_r", "scalars": []}',
        "RESULT:",
    ],
)
def test_an_unreadable_result_line_is_a_crash(
    line: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_child(monkeypatch, f"print({line!r})\n")
    out = run_example(tmp_path, "pandas", TIMEOUT)

    assert (out.ok, out.error_type) == (False, "Crash")
    assert out.error_message is not None
    assert out.error_message.startswith("exit 0: ")
    assert out.datasets == []
    assert out.scalars == []


def test_a_child_that_cannot_start_is_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def command(path: Path, engine: Engine, repeat: int) -> list[str]:
        return [str(tmp_path / "no-such-python")]

    monkeypatch.setattr(runner, "_child_command", command)
    out = run_example(tmp_path, "pandas", TIMEOUT)

    assert (out.ok, out.error_type) == (False, "Crash")
    assert out.error_message is not None
    assert out.error_message.startswith("cannot start")


def test_a_hanging_child_is_killed_and_started_once_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = fake_child(monkeypatch, HANG)
    killed: list[int] = []
    kill_group = runner._kill_group

    def spy(process: subprocess.Popen[Any]) -> None:
        killed.append(process.pid)
        kill_group(process)

    monkeypatch.setattr(runner, "_kill_group", spy)

    began = time.monotonic()
    out = run_example(tmp_path, "pandas", timeout_s=0.3)

    assert time.monotonic() - began < 10
    assert (out.ok, out.error_type, out.error_code) == (False, "Timeout", None)
    assert out.error_message is not None
    assert "0.3" in out.error_message
    assert len(started) == 2
    assert len(killed) == 2
    assert len(set(killed)) == 2


def test_a_run_that_times_out_once_can_still_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = fake_child(monkeypatch, HANG, printing(OK_PAYLOAD))

    # The second child has the same limit as the first: a second is plenty for a print.
    assert run_example(tmp_path, "pandas", timeout_s=1.0) == OK_OUTPUT
    assert len(started) == 2


def test_an_interrupt_kills_the_child_that_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = fake_child(monkeypatch, HANG)
    exit_codes: list[int] = []
    kill_group = runner._kill_group

    def spy(process: subprocess.Popen[Any]) -> None:
        kill_group(process)
        exit_codes.append(process.wait(timeout=10))  # it is gone

    def interrupted(self: subprocess.Popen[str], *args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_kill_group", spy)
    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_example(tmp_path, "pandas", TIMEOUT)

    assert exit_codes == [-signal.SIGKILL]
    assert len(started) == 1  # an interrupt is not a timeout: there is no second attempt


def test_a_crash_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = fake_child(monkeypatch, "import sys\nsys.exit(1)\n")

    assert run_example(tmp_path, "pandas", TIMEOUT).error_type == "Crash"
    assert len(started) == 1


def test_killing_the_group_kills_the_children_of_the_child_too(tmp_path: Path) -> None:
    beat = tmp_path / "beat.py"
    beat.write_text(
        dedent(
            """\
            import sys, time

            for _ in range(1000):
                with open(sys.argv[1], "a") as handle:
                    handle.write("x")
                time.sleep(0.02)
            """
        ),
        encoding="utf-8",
    )
    # The child starts a process that keeps writing to a file, and waits until it does.
    parent = dedent(
        """\
        import os, subprocess, sys, time

        subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
        for _ in range(500):
            if os.path.exists(sys.argv[2]):
                break
            time.sleep(0.01)
        print("ready", flush=True)
        time.sleep(60)
        """
    )
    heartbeat = tmp_path / "heartbeat"
    process = process_of_its_own(parent, str(beat), str(heartbeat))
    assert process.stdout is not None
    with process:
        try:
            assert process.stdout.readline().strip() == "ready"
            alive = heartbeat.stat().st_size
            time.sleep(0.2)
            assert heartbeat.stat().st_size > alive  # the process of the child is writing

            runner._kill_group(process)
            process.wait(timeout=10)
            final = heartbeat.stat().st_size
            time.sleep(0.3)
            assert heartbeat.stat().st_size == final  # and now it is not
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)  # do not leave it writing if the check failed
            raise


def test_killing_a_group_that_is_gone_is_not_an_error() -> None:
    process = process_of_its_own("pass")
    with process:
        process.wait(timeout=10)

    runner._kill_group(process)  # the child ended, and with it its group


def test_run_all_returns_the_result_of_every_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    echo = dedent(
        """\
        import json, sys

        path, engine, repeat = sys.argv[1:]
        value = "|".join([path, engine, repeat])
        scalar = {"name": "echo", "type": "String", "value": value}
        print("RESULT:" + json.dumps({"ok": True, "datasets": [], "scalars": [scalar]}))
        """
    )
    started = fake_child(monkeypatch, echo)
    jobs: list[tuple[Path, Engine, int]] = [
        (tmp_path / "a", "pandas", 1),
        (tmp_path / "a", "duckdb", 1),
        (tmp_path / "b", "pandas", 2),
        (tmp_path / "c", "duckdb", 2),
        (tmp_path / "d", "pandas", 1),
    ]

    results = run_all(jobs, workers=3, timeout_s=TIMEOUT)

    assert list(results) == [(path, engine) for path, engine, _ in jobs]  # the order of the jobs
    for path, engine, repeat in jobs:
        out = results[(path, engine)]
        assert out.ok
        assert out.stable is None
        assert out.scalars == [ActualScalar("echo", "String", f"{path}|{engine}|{repeat}")]
    assert len(started) == len(jobs)


def test_run_all_runs_the_jobs_side_by_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each job leaves a file and waits for the other two: run one after the other, a job would
    # see only the files of the jobs before it.
    rendezvous = dedent(
        """\
        import json, os, sys, time

        folder, name = os.path.split(sys.argv[1])
        open(os.path.join(folder, name), "w").close()
        for _ in range(100):
            seen = len(os.listdir(folder))
            if seen == 3:
                break
            time.sleep(0.05)
        scalar = {"name": "seen", "type": "Integer", "value": str(seen)}
        print("RESULT:" + json.dumps({"ok": True, "datasets": [], "scalars": [scalar]}))
        """
    )
    fake_child(monkeypatch, rendezvous)
    folder = tmp_path / "jobs"
    folder.mkdir()
    jobs: list[tuple[Path, Engine, int]] = [(folder / name, "pandas", 1) for name in "abc"]

    results = run_all(jobs, workers=3, timeout_s=TIMEOUT)

    for out in results.values():
        assert out.scalars == [ActualScalar("seen", "Integer", "3")]


def test_run_all_applies_the_timeout_to_every_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_child(monkeypatch, HANG)
    jobs: list[tuple[Path, Engine, int]] = [
        (tmp_path / "a", "pandas", 1),
        (tmp_path / "b", "duckdb", 2),
    ]

    began = time.monotonic()
    results = run_all(jobs, workers=2, timeout_s=0.3)

    assert time.monotonic() - began < 10
    assert [(out.ok, out.error_type) for out in results.values()] == [(False, "Timeout")] * 2


def test_run_all_cancels_the_jobs_that_have_not_started_when_one_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[Path] = []

    def stub(
        path: Path, engine: Engine, timeout_s: float, repeat: int = 1, **_: object
    ) -> RunOutput:
        started.append(path)
        if len(started) == 1:
            raise RuntimeError("interrupted")
        time.sleep(0.05)
        return OK_OUTPUT

    monkeypatch.setattr(runner, "run_example", stub)
    jobs: list[tuple[Path, Engine, int]] = [(tmp_path / str(n), "pandas", 1) for n in range(40)]

    with pytest.raises(RuntimeError, match="interrupted"):
        run_all(jobs, workers=1, timeout_s=TIMEOUT)

    assert len(started) < 10  # the first one, and what the worker took before the cancel


def test_run_all_with_no_jobs() -> None:
    assert run_all([], workers=2, timeout_s=TIMEOUT) == {}


def test_an_interrupt_of_run_all_kills_its_children_at_once_and_starts_no_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each child writes a line when it starts, then hangs. Three of the six jobs run at a time,
    # and the interrupt comes to the main thread, as a Ctrl-C does, once the three run.
    log = tmp_path / "started.log"
    hang = dedent(
        f"""\
        import sys, time

        with open({str(log)!r}, "a") as handle:
            handle.write(sys.argv[1] + "\\n")
        time.sleep(60)
        """
    )
    fake_child(monkeypatch, hang)

    def started() -> int:
        return len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0

    sent: list[float] = []

    def interrupt_when_three_run() -> None:
        deadline = time.monotonic() + 30
        while started() < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        if started() == 3:  # never after run_all is over: the children hang far longer
            sent.append(time.monotonic())
            signal.pthread_kill(threading.main_thread().ident or 0, signal.SIGINT)

    jobs: list[tuple[Path, Engine, int]] = [(tmp_path / str(n), "pandas", 1) for n in range(6)]
    interrupter = threading.Thread(target=interrupt_when_three_run)
    interrupter.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            run_all(jobs, workers=3, timeout_s=3.0)
        surfaced = time.monotonic()
    finally:
        interrupter.join()

    assert len(sent) == 1
    time.sleep(0.5)
    waited, children = surfaced - sent[0], started()
    # The run does not wait for its children to time out (3 s each, then once more): it kills
    # them. And no child starts after the interrupt: no second attempt, no job that was waiting.
    assert (waited < 1.5, children) == (True, 3), f"{waited:.2f} s, {children} children"


# --- The preflight ---


def test_the_preflight_gives_the_versions_of_the_engines() -> None:
    # A real child: the one that every run needs, with the engines of this worktree.
    assert runner.preflight(TIMEOUT) == runner.EngineVersions(
        pandas=pd.__version__, duckdb=duckdb.__version__
    )


@pytest.mark.parametrize(
    "code",
    [
        "import sys\nsys.stderr.write('Traceback\\nModuleNotFoundError: No module named "
        "\\'vtlengine.AST.Grammar._cpp_parser.vtl_cpp_parser\\'\\n')\nsys.exit(1)\n",
        "print('PREFLIGHT:not json')\n",
        "print('nothing to say')\n",
    ],
    ids=["import error", "unreadable answer", "no answer"],
)
def test_a_preflight_child_that_fails_is_one_line_with_the_reason(
    code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_preflight_command", lambda: [sys.executable, "-c", code])
    with pytest.raises(runner.PreflightError) as failed:
        runner.preflight(TIMEOUT)
    message = str(failed.value)
    assert "\n" not in message
    assert message.startswith("preflight: a child process cannot import vtlengine and its engines")
    assert message.endswith(
        "run the generator with a Python that can import vtlengine: "
        "the compiled parser must be in src/"
    )
    if "ModuleNotFoundError" in code:
        assert (
            "(ModuleNotFoundError: No module named 'vtlengine.AST.Grammar._cpp_parser." in message
        )


def test_a_preflight_child_that_hangs_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_preflight_command", lambda: [sys.executable, "-c", HANG])
    began = time.monotonic()
    with pytest.raises(runner.PreflightError, match="no answer in 0.5 s"):
        runner.preflight(0.5)
    assert time.monotonic() - began < 10


# --- Environment of the child ---


def test_child_env_points_at_this_worktree() -> None:
    env = child_env()

    assert env["PYTHONPATH"] == os.pathsep.join(
        [str(WORKTREE / "src"), str(WORKTREE / "docs" / "operator_map")]
    )
    assert (WORKTREE / "src" / "vtlengine").is_dir()
    assert (WORKTREE / "docs" / "operator_map" / "tool").is_dir()
    assert env["VTL_THREADS"] == "1"
    assert env["VTL_MEMORY_LIMIT"] == "1GB"
    # The rest of the environment is the one of the process that starts the child.
    own = {"PYTHONPATH", "VTL_THREADS", "VTL_MEMORY_LIMIT"}
    assert {k: v for k, v in env.items() if k not in own} == {
        k: v for k, v in os.environ.items() if k not in own
    }


def test_child_env_is_a_new_dictionary_every_time() -> None:
    env = child_env()
    env["VTL_THREADS"] = "8"

    assert child_env()["VTL_THREADS"] == "1"
