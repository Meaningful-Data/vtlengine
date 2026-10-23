# ruff: noqa: S101
"""Checks for the verdicts of the engines, the status of an example and the roll-up of statuses.

The runs are small hand-made ``RunOutput`` values and the expected files small hand-made
``ExampleFiles``: no engine runs here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tool.compare import (
    ActualComponent,
    ActualDataset,
    ActualScalar,
    compare_dataset,
    compare_scalar,
)
from tool.examples import ExampleFiles, ExpectedComponent, ExpectedDataset, ExpectedScalar
from tool.model import Catalogue, Chapter, Operator, Status, TypeDef, Variant, authored, manual
from tool.runner import RunOutput
from tool.status import (
    EMOJI,
    EngineVerdict,
    ExampleResult,
    StatusIndex,
    build_index,
    counts,
    example_result,
    rollup,
    verdict,
)

TODAY = "2026-10-23"


# --- Builders ---


def row(id_1: str, me_1: str | None) -> dict[str, str | None]:
    return {"Id_1": id_1, "Me_1": me_1}


def expected_dataset(
    rows: list[dict[str, str | None]] | None = None, name: str = "DS_r", measure: str = "Number"
) -> ExpectedDataset:
    return ExpectedDataset(
        name,
        {
            "Id_1": ExpectedComponent("Identifier", ("Integer",), False),
            "Me_1": ExpectedComponent("Measure", (measure,), None),
        },
        [row("1", "3.5"), row("2", None)] if rows is None else rows,
    )


def actual_dataset(
    rows: list[dict[str, str | None]] | None = None, name: str = "DS_r"
) -> ActualDataset:
    return ActualDataset(
        name,
        {
            "Id_1": ActualComponent("Identifier", "Integer", False),
            "Me_1": ActualComponent("Measure", "Number", True),
        },
        [row("1", "3.5"), row("2", None)] if rows is None else rows,
    )


def files(
    datasets: list[ExpectedDataset] | None = None, scalars: list[ExpectedScalar] | None = None
) -> ExampleFiles:
    """The files of an example; by default it expects the dataset ``DS_r``."""
    return ExampleFiles(
        path=Path("example"),
        script="DS_r := DS_1;",
        structures=[],
        datapoints={},
        value_domains=[],
        external_routines=[],
        expected_datasets=[expected_dataset()] if datasets is None else datasets,
        expected_scalars=[] if scalars is None else scalars,
    )


def run(*outputs: ActualDataset | ActualScalar, stable: bool | None = None) -> RunOutput:
    """A run that worked and gave ``outputs``."""
    return RunOutput(
        True,
        [output for output in outputs if isinstance(output, ActualDataset)],
        [output for output in outputs if isinstance(output, ActualScalar)],
        stable=stable,
    )


def failed(
    error_type: str | None = "SemanticError",
    error_code: str | None = "1-1-1-1",
    error_message: str | None = "no such operator",
) -> RunOutput:
    """A run that raised an error."""
    return RunOutput(False, [], [], error_type, error_code, error_message)


def numbers(*values: str | None, name: str = "DS_r") -> ActualDataset:
    """A dataset with the identifiers 1, 2... whose measure is ``values``."""
    return actual_dataset([row(str(n), value) for n, value in enumerate(values, start=1)], name)


def random_files() -> ExampleFiles:
    """A ``random`` example expects a dataset but not its values: the expected file is empty."""
    return files(datasets=[expected_dataset(rows=[])])


def date_scalar(value: str | None, name: str = "sc_r") -> ActualScalar:
    return ActualScalar(name, "Date", value)


def date_files() -> ExampleFiles:
    """A ``current_date`` example expects a Date scalar but not its value."""
    return files(datasets=[], scalars=[ExpectedScalar("sc_r", "Date", None)])


def date_dataset_files() -> ExampleFiles:
    """A ``current_date`` example that expects a dataset with a Date measure, without values."""
    return files(datasets=[expected_dataset(rows=[], measure="Date")])


def dates(*values: str | None) -> ActualDataset:
    """A dataset with the identifiers 1, 2... whose measure is a Date, with ``values``."""
    components = {
        "Id_1": ActualComponent("Identifier", "Integer", False),
        "Me_1": ActualComponent("Measure", "Date", True),
    }
    return ActualDataset(
        "DS_r", components, [row(str(n), value) for n, value in enumerate(values, start=1)]
    )


def result(status: Status) -> ExampleResult:
    """A result with ``status``: the index looks at nothing else."""
    matched = EngineVerdict("match", "")
    return ExampleResult(matched, matched, True, status)


# --- Roll-up and counts ---


def test_emoji_by_status() -> None:
    assert EMOJI == {"ok": "🟢", "partial": "🟡", "missing": "🔴", "diverge": "🔵"}


def test_rollup_precedence() -> None:
    assert rollup(["ok", "diverge", "missing"]) == "diverge"
    assert rollup(["ok", "ok"]) == "ok"
    assert rollup(["missing", "missing"]) == "missing"
    assert rollup(["ok", "missing"]) == "partial"


def test_rollup_of_the_other_mixes() -> None:
    assert rollup(["ok"]) == "ok"
    assert rollup(["partial"]) == "partial"
    assert rollup(["ok", "partial"]) == "partial"
    assert rollup(["missing", "partial"]) == "partial"
    assert rollup(["diverge"]) == "diverge"
    assert rollup(["diverge", "ok", "ok"]) == "diverge"  # wherever the divergence is
    assert rollup(["ok", "ok", "diverge"]) == "diverge"
    assert rollup(["missing", "diverge", "partial"]) == "diverge"


def test_rollup_of_nothing_raises() -> None:
    with pytest.raises(ValueError, match="nothing to roll up"):
        rollup([])


def test_counts_has_every_status_in_order() -> None:
    tally = counts(["ok", "diverge", "ok", "missing"])
    assert tally == {"ok": 2, "partial": 0, "missing": 1, "diverge": 1}
    assert list(tally) == ["ok", "partial", "missing", "diverge"]
    assert list(counts([]).values()) == [0, 0, 0, 0]
    assert list(counts([]).keys()) == ["ok", "partial", "missing", "diverge"]


# --- Verdicts: exact ---


def test_exact_output_that_matches() -> None:
    got = verdict(authored(), files(), run(actual_dataset()), TODAY)
    assert got == EngineVerdict("match", "")


def test_exact_mismatch_has_the_first_problem_as_detail() -> None:
    wrong = actual_dataset([row("1", "9"), row("2", None)])
    problem = compare_dataset(expected_dataset(), wrong)
    assert problem is not None
    got = verdict(authored(), files(), run(wrong), TODAY)
    assert got == EngineVerdict("mismatch", problem)


def test_exact_scalars_are_compared() -> None:
    wanted = files(datasets=[], scalars=[ExpectedScalar("sc_r", "Integer", "8")])
    assert verdict(authored(), wanted, run(ActualScalar("sc_r", "Integer", "8")), TODAY) == (
        EngineVerdict("match", "")
    )
    wrong = ActualScalar("sc_r", "Integer", "9")
    problem = compare_scalar(ExpectedScalar("sc_r", "Integer", "8"), wrong)
    assert problem is not None
    assert verdict(authored(), wanted, run(wrong), TODAY) == EngineVerdict("mismatch", problem)


def test_every_expected_output_must_match() -> None:
    wanted = files(datasets=[expected_dataset()], scalars=[ExpectedScalar("sc_r", "Integer", "8")])
    good = ActualScalar("sc_r", "Integer", "8")
    assert verdict(authored(), wanted, run(actual_dataset(), good), TODAY).outcome == "match"
    bad_dataset = actual_dataset([row("1", "9"), row("2", None)])
    assert verdict(authored(), wanted, run(bad_dataset, good), TODAY).outcome == "mismatch"
    bad_scalar = ActualScalar("sc_r", "Integer", "9")
    assert verdict(authored(), wanted, run(actual_dataset(), bad_scalar), TODAY).outcome == (
        "mismatch"
    )


def test_outputs_that_the_expected_files_do_not_name_are_ignored() -> None:
    rubbish = actual_dataset([row("7", "0")], name="DS_tmp")
    out = run(actual_dataset(), rubbish, ActualScalar("sc_tmp", "Integer", "1"))
    assert verdict(authored(), files(), out, TODAY) == EngineVerdict("match", "")


@pytest.mark.parametrize("check", ["exact", "random", "current_date"])
def test_an_expected_output_missing_from_the_run_is_a_mismatch(check: str) -> None:
    example = authored(check=check)
    other = run(actual_dataset(name="DS_other"), stable=True)
    assert verdict(example, files(), other, TODAY) == (
        EngineVerdict("mismatch", "missing output DS_r")
    )
    assert verdict(example, files(), run(stable=True), TODAY) == (
        EngineVerdict("mismatch", "missing output DS_r")
    )
    wanted = files(datasets=[], scalars=[ExpectedScalar("sc_r", "Integer", "8")])
    assert verdict(example, wanted, other, TODAY) == (
        EngineVerdict("mismatch", "missing output sc_r")
    )


def test_the_detail_of_a_mismatch_is_the_first_problem_in_file_order() -> None:
    wanted = files(datasets=[expected_dataset(), expected_dataset(name="DS_s")])
    first = actual_dataset([row("1", "9"), row("2", None)])
    second = actual_dataset([row("1", "8"), row("2", None)], name="DS_s")
    first_problem = compare_dataset(expected_dataset(), first)
    second_problem = compare_dataset(expected_dataset(name="DS_s"), second)
    assert first_problem is not None
    assert second_problem is not None
    assert first_problem != second_problem
    # An example with several expected outputs names the one that is wrong.
    both = verdict(authored(), wanted, run(first, second), TODAY)
    assert both == EngineVerdict("mismatch", f"DS_r: {first_problem}")
    only_second = verdict(authored(), wanted, run(actual_dataset(), second), TODAY)
    assert only_second == EngineVerdict("mismatch", f"DS_s: {second_problem}")


def test_the_detail_names_the_output_only_when_the_example_expects_several() -> None:
    wrong_scalar = ActualScalar("sc_r", "Integer", "9")
    scalar_problem = compare_scalar(ExpectedScalar("sc_r", "Integer", "8"), wrong_scalar)
    assert scalar_problem is not None
    # A dataset and a scalar: the scalar is the one that is wrong, and the detail says so.
    several = files(datasets=[expected_dataset()], scalars=[ExpectedScalar("sc_r", "Integer", "8")])
    got = verdict(authored(), several, run(actual_dataset(), wrong_scalar), TODAY)
    assert got == EngineVerdict("mismatch", f"sc_r: {scalar_problem}")
    # The structure check of the property checks names the output too.
    properties = files(
        datasets=[expected_dataset(rows=[])], scalars=[ExpectedScalar("sc_r", "Number", None)]
    )
    out = run(numbers("0.25"), ActualScalar("sc_r", "Integer", "0.5"), stable=True)
    assert verdict(authored(check="random"), properties, out, TODAY) == (
        EngineVerdict("mismatch", "sc_r: type expected Number got Integer")
    )
    # One expected output needs no name: there is nothing to tell apart.
    single = files(datasets=[], scalars=[ExpectedScalar("sc_r", "Integer", "8")])
    got = verdict(authored(), single, run(wrong_scalar), TODAY)
    assert got == EngineVerdict("mismatch", scalar_problem)


def test_a_scalar_does_not_stand_for_an_expected_dataset() -> None:
    out = run(ActualScalar("DS_r", "Integer", "1"))
    assert verdict(authored(), files(), out, TODAY) == (
        EngineVerdict("mismatch", "missing output DS_r")
    )


# --- Verdicts: errors ---


def test_error_detail_is_type_code_and_message() -> None:
    got = verdict(authored(), files(), failed("SemanticError", "1-1-1-1", "no operator"), TODAY)
    assert got == EngineVerdict("error", "SemanticError 1-1-1-1: no operator")


@pytest.mark.parametrize(
    ("error_type", "error_code", "error_message", "detail"),
    [
        (
            "Timeout",
            None,
            "no result in 120 s, 2 attempts",
            "Timeout: no result in 120 s, 2 attempts",
        ),
        ("SemanticError", "1-1-1-1", None, "SemanticError 1-1-1-1"),
        ("Crash", None, None, "Crash"),
        (None, "1-1-1-1", "no operator", "1-1-1-1: no operator"),
        (None, None, "no operator", "no operator"),
        (None, "1-1-1-1", None, "1-1-1-1"),
        ("Crash", "", "", "Crash"),
        (None, None, None, ""),
    ],
)
def test_error_detail_leaves_out_what_is_missing(
    error_type: str | None, error_code: str | None, error_message: str | None, detail: str
) -> None:
    out = failed(error_type, error_code, error_message)
    got = verdict(authored(), files(), out, TODAY)
    assert got == EngineVerdict("error", detail)
    assert "None" not in got.detail


def test_a_failed_run_is_an_error_whatever_the_check() -> None:
    for check in ("exact", "random", "current_date"):
        got = verdict(authored(check=check), files(), failed("Timeout", None, "late"), TODAY)
        assert got == EngineVerdict("error", "Timeout: late")


# --- Verdicts: random ---


def test_random_values_in_the_unit_interval_that_are_stable_match() -> None:
    out = run(numbers("0.708194", "0", "0.999999", None), stable=True)
    assert verdict(authored(check="random"), random_files(), out, TODAY) == (
        EngineVerdict("match", "")
    )


def test_random_scalars_are_checked_too() -> None:
    wanted = files(datasets=[], scalars=[ExpectedScalar("sc_r", "Number", None)])
    good = run(ActualScalar("sc_r", "Number", "5.5e-05"), stable=True)
    assert verdict(authored(check="random"), wanted, good, TODAY).outcome == "match"
    bad = run(ActualScalar("sc_r", "Number", "1.5"), stable=True)
    assert verdict(authored(check="random"), wanted, bad, TODAY).outcome == "mismatch"


def test_random_ignores_identifiers_and_outputs_that_are_not_expected() -> None:
    # The identifiers 1 and 2 are not in [0, 1), and neither is the dataset the files do not name.
    out = run(numbers("0.25", "0.5"), numbers("7", name="DS_tmp"), stable=True)
    assert verdict(authored(check="random"), random_files(), out, TODAY).outcome == "match"


@pytest.mark.parametrize(
    "value", ["1", "1.0", "1.5", "-0.1", "abc", "", "NaN", "inf", "0_5", "1e0"]
)
def test_random_value_outside_the_unit_interval_is_a_mismatch(value: str) -> None:
    out = run(numbers("0.5", value), stable=True)
    got = verdict(authored(check="random"), random_files(), out, TODAY)
    assert got.outcome == "mismatch"
    assert repr(value) in got.detail


@pytest.mark.parametrize("stable", [False, None])
def test_random_needs_a_stable_run(stable: bool | None) -> None:
    out = run(numbers("0.5", "0.25"), stable=stable)
    got = verdict(authored(check="random"), random_files(), out, TODAY)
    assert got.outcome == "mismatch"
    assert got.detail


def test_random_needs_at_least_one_value() -> None:
    # The dataset is there and the run is stable: the lack of a value is the only problem.
    wanted = random_files()
    for out in (
        run(numbers(), stable=True),  # no data points
        run(numbers(None, None), stable=True),  # only nulls
    ):
        assert verdict(authored(check="random"), wanted, out, TODAY).outcome == "mismatch"


# --- Verdicts: current_date ---


def test_current_date_scalar_that_is_today_matches() -> None:
    out = run(date_scalar(TODAY))
    assert verdict(authored(check="current_date"), date_files(), out, TODAY) == (
        EngineVerdict("match", "")
    )


def test_current_date_in_a_dataset_matches_and_nulls_are_skipped() -> None:
    out = run(dates(TODAY, None, TODAY))
    got = verdict(authored(check="current_date"), date_dataset_files(), out, TODAY)
    assert got.outcome == "match"


def test_current_date_that_is_another_day_is_a_mismatch() -> None:
    other = run(date_scalar("2026-10-22"))
    got = verdict(authored(check="current_date"), date_files(), other, TODAY)
    assert got.outcome == "mismatch"
    assert "2026-10-22" in got.detail
    one_bad = run(dates(TODAY, "2020-01-31"))
    got = verdict(authored(check="current_date"), date_dataset_files(), one_bad, TODAY)
    assert got.outcome == "mismatch"
    assert "2020-01-31" in got.detail


def test_current_date_needs_at_least_one_date_value() -> None:
    # The structure is the expected one and the outputs are there: the lack of a Date value is
    # the only problem. A value that is not of the type Date does not count.
    check = authored(check="current_date")
    assert verdict(check, date_files(), run(date_scalar(None)), TODAY).outcome == "mismatch"
    string_files = files(datasets=[], scalars=[ExpectedScalar("sc_r", "String", None)])
    not_a_date = run(ActualScalar("sc_r", "String", TODAY))
    assert verdict(check, string_files, not_a_date, TODAY).outcome == "mismatch"
    assert verdict(check, date_dataset_files(), run(dates(None, None)), TODAY).outcome == (
        "mismatch"
    )
    number_files = files(datasets=[expected_dataset(rows=[])])
    assert verdict(check, number_files, run(numbers("0.5")), TODAY).outcome == "mismatch"


def test_current_date_ignores_outputs_that_are_not_expected() -> None:
    out = run(date_scalar(TODAY), date_scalar("1999-12-31", name="sc_old"))
    assert verdict(authored(check="current_date"), date_files(), out, TODAY).outcome == "match"


# --- Verdicts: the structure of the property checks ---

IDENTIFIER = ActualComponent("Identifier", "Integer", False)


@pytest.mark.parametrize(
    ("components", "problem"),
    [
        (
            {"Id_1": IDENTIFIER, "Me_1": ActualComponent("Measure", "String", True)},
            "type of Me_1 expected Number got String",
        ),
        (
            {"Id_1": IDENTIFIER, "Me_1": ActualComponent("Attribute", "Number", True)},
            "role of Me_1 expected Measure got Attribute",
        ),
        ({"Id_1": IDENTIFIER}, "missing components: Me_1"),
        (
            {
                "Id_1": IDENTIFIER,
                "Me_1": ActualComponent("Measure", "Number", True),
                "Me_2": ActualComponent("Measure", "Number", True),
            },
            "unexpected components: Me_2",
        ),
    ],
    ids=["type", "role", "missing", "unexpected"],
)
def test_random_needs_the_expected_structure(
    components: dict[str, ActualComponent], problem: str
) -> None:
    # The run is stable and the values are numbers in [0, 1), but for a missing Me_1, which
    # leaves no value at all. The structure is judged first, with the rules of compare_dataset.
    wrong = ActualDataset("DS_r", components, [row("1", "0.25"), row("2", "0.5")])
    got = verdict(authored(check="random"), random_files(), run(wrong, stable=True), TODAY)
    assert got == EngineVerdict("mismatch", problem)


def test_random_needs_the_expected_type_of_a_scalar() -> None:
    wanted = files(datasets=[], scalars=[ExpectedScalar("sc_r", "Number", None)])
    out = run(ActualScalar("sc_r", "Integer", "0.5"), stable=True)
    assert verdict(authored(check="random"), wanted, out, TODAY) == (
        EngineVerdict("mismatch", "type expected Number got Integer")
    )


def test_current_date_needs_the_expected_structure() -> None:
    check = authored(check="current_date")
    # A String that is today is not the Date that the example expects.
    out = run(ActualScalar("sc_r", "String", TODAY))
    assert verdict(check, date_files(), out, TODAY) == (
        EngineVerdict("mismatch", "type expected Date got String")
    )
    # Dates that are today, in a dataset that lacks its identifier.
    components = {"Me_1": ActualComponent("Measure", "Date", True)}
    dated = ActualDataset("DS_r", components, [{"Me_1": TODAY}])
    assert verdict(check, date_dataset_files(), run(dated), TODAY) == (
        EngineVerdict("mismatch", "missing components: Id_1")
    )


def test_an_unknown_check_is_an_error_of_the_catalogue() -> None:
    with pytest.raises(ValueError, match="unknown check 'randm'"):
        verdict(authored(check="randm"), files(), run(actual_dataset()), TODAY)


# --- Example results ---


def test_both_engines_matching_is_ok() -> None:
    out = run(actual_dataset())
    got = example_result(authored(), files(), out, out, TODAY)
    assert got == ExampleResult(EngineVerdict("match", ""), EngineVerdict("match", ""), True, "ok")


def test_random_values_must_match_across_engines() -> None:
    pandas = run(numbers("0.708194"), stable=True)
    duckdb = run(numbers("0.195471"), stable=True)
    got = example_result(authored(check="random"), random_files(), pandas, duckdb, TODAY)
    assert got.pandas.outcome == "match"  # each passes the property
    assert got.duckdb.outcome == "match"
    assert not got.engines_agree  # but they are not the same values
    assert got.status == "diverge"


def test_random_values_that_are_the_same_on_both_engines_are_ok() -> None:
    out = run(numbers("0.708194"), stable=True)
    got = example_result(authored(check="random"), random_files(), out, out, TODAY)
    assert got.engines_agree
    assert got.status == "ok"


def test_current_date_status_follows_the_dates_of_the_engines() -> None:
    out = run(date_scalar(TODAY))
    got = example_result(authored(check="current_date"), date_files(), out, out, TODAY)
    assert got.status == "ok"
    yesterday = run(date_scalar("2026-10-22"))
    got = example_result(authored(check="current_date"), date_files(), out, yesterday, TODAY)
    assert got.status == "diverge"


def test_same_wrong_answer_on_both_is_partial() -> None:
    wrong = run(actual_dataset([row("1", "9"), row("2", None)]))
    got = example_result(authored(), files(), wrong, wrong, TODAY)
    assert got.pandas.outcome == "mismatch"
    assert got.duckdb.outcome == "mismatch"
    assert got.engines_agree
    assert got.status == "partial"

    other = run(actual_dataset([row("1", "8"), row("2", None)]))
    differing = example_result(authored(), files(), wrong, other, TODAY)
    assert differing.pandas.outcome == differing.duckdb.outcome == "mismatch"
    assert not differing.engines_agree
    assert differing.status == "diverge"


def test_both_errors_are_missing_even_if_messages_differ() -> None:
    pandas = failed("SemanticError", "1-1-1-1", "no such operator")
    duckdb = failed("RunTimeError", "2-3-6", "cannot cast")
    got = example_result(authored(), files(), pandas, duckdb, TODAY)
    assert got.pandas.outcome == got.duckdb.outcome == "error"
    assert got.pandas.detail != got.duckdb.detail
    assert got.engines_agree
    assert got.status == "missing"


@pytest.mark.parametrize("pandas_first", [True, False])
@pytest.mark.parametrize(
    ("one", "other"),
    [
        (run(actual_dataset()), run(actual_dataset([row("1", "9"), row("2", None)]))),
        (run(actual_dataset()), failed()),
        (run(actual_dataset([row("1", "9"), row("2", None)])), failed()),
    ],
    ids=["match-mismatch", "match-error", "mismatch-error"],
)
def test_different_outcomes_diverge(one: RunOutput, other: RunOutput, pandas_first: bool) -> None:
    pandas, duckdb = (one, other) if pandas_first else (other, one)
    got = example_result(authored(), files(), pandas, duckdb, TODAY)
    assert got.pandas.outcome != got.duckdb.outcome
    assert not got.engines_agree
    assert got.status == "diverge"


def test_engines_that_both_match_but_not_each_other_diverge() -> None:
    # The expected 3.5 has one decimal, so 3.52 and 3.54 both match it. The engines differ.
    pandas = run(actual_dataset([row("1", "3.52"), row("2", None)]))
    duckdb = run(actual_dataset([row("1", "3.54"), row("2", None)]))
    got = example_result(authored(), files(), pandas, duckdb, TODAY)
    assert got.pandas.outcome == got.duckdb.outcome == "match"
    assert not got.engines_agree
    assert got.status == "diverge"


def test_intermediate_outputs_do_not_decide_whether_the_engines_agree() -> None:
    pandas = run(actual_dataset(), actual_dataset([row("1", "1")], name="DS_tmp"))
    duckdb = run(actual_dataset(), actual_dataset([row("1", "2")], name="DS_tmp"))
    got = example_result(authored(), files(), pandas, duckdb, TODAY)
    assert got.engines_agree
    assert got.status == "ok"


def test_engines_that_each_miss_a_different_output_diverge() -> None:
    wanted = files(
        datasets=[expected_dataset(), expected_dataset(name="DS_s")],
    )
    pandas = run(actual_dataset())  # lacks DS_s
    duckdb = run(actual_dataset(name="DS_s"))  # lacks DS_r
    got = example_result(authored(), wanted, pandas, duckdb, TODAY)
    assert got.pandas == EngineVerdict("mismatch", "missing output DS_s")
    assert got.duckdb == EngineVerdict("mismatch", "missing output DS_r")
    assert not got.engines_agree
    assert got.status == "diverge"


def test_scalars_and_datasets_are_compared_together_between_engines() -> None:
    wanted = files(scalars=[ExpectedScalar("sc_r", "Integer", "8")])
    pandas = run(actual_dataset(), ActualScalar("sc_r", "Integer", "8"))
    duckdb = run(actual_dataset(), ActualScalar("sc_r", "Integer", "8"))
    assert example_result(authored(), wanted, pandas, duckdb, TODAY).status == "ok"
    other = run(actual_dataset(), ActualScalar("sc_r", "Integer", "9"))
    got = example_result(authored(), wanted, pandas, other, TODAY)
    assert got.duckdb.outcome == "mismatch"
    assert got.status == "diverge"


# --- Index ---


def addition() -> Operator:
    return Operator(
        id="addition",
        name="Addition",
        symbol="+",
        page="Numeric Operators/Addition",
        accepts=("number",),
        returns=("number",),
        variants=(
            Variant("ds_ds", "DS + DS", (manual("ex_1"), authored())),
            Variant("ds_scalar", "DS + scalar", (authored(),)),
        ),
    )


def subtraction() -> Operator:
    return Operator(
        id="subtraction",
        name="Subtraction",
        symbol="-",
        page="Numeric Operators/Subtraction",
        accepts=("number",),
        returns=("number",),
        variants=(Variant("ds_ds", "DS - DS", (authored(),)),),
    )


def numeric(*operators: Operator) -> Chapter:
    return Chapter("numeric", "Numeric Operators", operators)


def addition_results(changes: dict[str, Status] | None = None) -> dict[str, ExampleResult]:
    """A result for every example of ``addition``: ok, but for the statuses in ``changes``."""
    statuses: dict[str, Status] = {
        "numeric/addition/ds_ds/manual_ex_1": "ok",
        "numeric/addition/ds_ds/minimal": "ok",
        "numeric/addition/ds_scalar/minimal": "ok",
        **(changes or {}),
    }
    return {key: result(status) for key, status in statuses.items()}


def test_build_index_keys() -> None:
    cat = Catalogue((numeric(addition()),), ())
    index = build_index(cat, addition_results())
    assert isinstance(index, StatusIndex)
    assert index.operators["numeric/addition"] == "ok"
    assert list(index.variants) == ["numeric/addition/ds_ds", "numeric/addition/ds_scalar"]
    assert list(index.examples) == [
        "numeric/addition/ds_ds/manual_ex_1",
        "numeric/addition/ds_ds/minimal",
        "numeric/addition/ds_scalar/minimal",
    ]
    assert list(index.operators) == ["numeric/addition"]
    assert index.chapters == {"numeric": "ok"}


def test_build_index_examples_are_the_results() -> None:
    cat = Catalogue((numeric(addition()),), ())
    results = addition_results({"numeric/addition/ds_ds/minimal": "partial"})
    index = build_index(cat, results)
    assert (
        index.examples["numeric/addition/ds_ds/minimal"]
        is results["numeric/addition/ds_ds/minimal"]
    )


def test_build_index_rolls_the_statuses_up() -> None:
    cat = Catalogue((numeric(addition(), subtraction()),), ())
    results = {
        **addition_results({"numeric/addition/ds_ds/minimal": "missing"}),
        "numeric/subtraction/ds_ds/minimal": result("missing"),
    }
    index = build_index(cat, results)
    assert index.variants == {
        "numeric/addition/ds_ds": "partial",  # ok and missing
        "numeric/addition/ds_scalar": "ok",
        "numeric/subtraction/ds_ds": "missing",
    }
    assert index.operators == {"numeric/addition": "partial", "numeric/subtraction": "missing"}
    assert index.chapters == {"numeric": "partial"}

    divergent = {**results, "numeric/subtraction/ds_ds/minimal": result("diverge")}
    index = build_index(cat, divergent)
    assert index.operators["numeric/subtraction"] == "diverge"
    assert index.chapters["numeric"] == "diverge"


def test_build_index_names_the_first_example_without_a_result() -> None:
    cat = Catalogue((numeric(addition(), subtraction()),), ())
    results = addition_results()
    with pytest.raises(ValueError, match="numeric/subtraction/ds_ds/minimal"):
        build_index(cat, results)
    del results["numeric/addition/ds_ds/minimal"]
    del results["numeric/addition/ds_scalar/minimal"]
    with pytest.raises(ValueError, match="numeric/addition/ds_ds/minimal") as info:
        build_index(cat, results)
    assert "ds_scalar" not in str(info.value)  # the first one in the catalogue, not all of them


def test_build_index_ignores_results_of_other_examples() -> None:
    cat = Catalogue((numeric(addition()),), ())
    results = {
        **addition_results(),
        "numeric/addition/ds_ds/removed": result("diverge"),
        "string/length/ds/minimal": result("missing"),
    }
    index = build_index(cat, results)
    assert "numeric/addition/ds_ds/removed" not in index.examples
    assert "string/length/ds/minimal" not in index.examples
    assert index.chapters == {"numeric": "ok"}
    assert index.operators == {"numeric/addition": "ok"}


def test_build_index_has_the_types_under_their_own_keys() -> None:
    types = (
        TypeDef("scalar", ()),  # a hub without examples
        TypeDef("number", ("scalar",), (Variant("identity", "identity", (authored(),)),)),
        TypeDef(
            "integer",
            ("number",),
            (
                Variant("identity", "identity", (authored(),)),
                Variant("cast", "cast", (authored(), authored("minimal_2"))),
            ),
        ),
    )
    cat = Catalogue((numeric(addition()),), types)
    results = {
        **addition_results(),
        "types/number/identity/minimal": result("ok"),
        "types/integer/identity/minimal": result("ok"),
        "types/integer/cast/minimal": result("missing"),
        "types/integer/cast/minimal_2": result("ok"),
    }
    index = build_index(cat, results)
    assert index.examples["types/integer/cast/minimal_2"] is results["types/integer/cast/minimal_2"]
    assert index.variants["types/integer/cast"] == "partial"
    assert index.operators["types/number"] == "ok"
    assert index.operators["types/integer"] == "partial"
    assert "types/scalar" not in index.operators  # no variants, no status
    assert index.chapters == {"numeric": "ok", "types": "partial"}

    missing = {
        key: value for key, value in results.items() if key != "types/number/identity/minimal"
    }
    with pytest.raises(ValueError, match="types/number/identity/minimal"):
        build_index(cat, missing)


def test_build_index_has_no_types_chapter_when_no_type_has_examples() -> None:
    cat = Catalogue((numeric(addition()),), (TypeDef("scalar", ()), TypeDef("number", ("scalar",))))
    index = build_index(cat, addition_results())
    assert index.chapters == {"numeric": "ok"}
    assert not any(key.startswith("types/") for key in index.operators)


@pytest.mark.parametrize(
    ("chapter", "key"),
    [
        (numeric(Operator("empty", "Empty", "e", "p", (), (), ())), "numeric/empty"),
        (
            numeric(Operator("empty", "Empty", "e", "p", (), (), (Variant("v", "v", ()),))),
            "numeric/empty/v",
        ),
        (numeric(), "numeric"),
    ],
    ids=["operator-without-variants", "variant-without-examples", "chapter-without-operators"],
)
def test_build_index_rejects_a_catalogue_entry_with_nothing_to_roll_up(
    chapter: Chapter, key: str
) -> None:
    with pytest.raises(ValueError, match=f"nothing to roll up for {key}"):
        build_index(Catalogue((chapter,), ()), {})


def test_status_values_are_frozen_dataclasses() -> None:
    index = build_index(Catalogue((numeric(addition()),), ()), addition_results())
    for value, field in (
        (EngineVerdict("match", ""), "detail"),
        (result("ok"), "status"),
        (index, "chapters"),
    ):
        assert dataclasses.is_dataclass(value)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, "other")
