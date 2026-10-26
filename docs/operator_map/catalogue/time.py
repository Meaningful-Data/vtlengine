"""The time operators of the VTL 2.2 Reference Manual.

The time series operators (``fill_time_series``, ``flow_to_stock``, ``stock_to_flow``,
``timeshift``) take a Data Set whose reference time Identifier is a *time*, a *date* or a
*time_period*: each of the three types is a variant. ``fill_time_series`` crosses the
*time_period* and *date* types with its limits method (absent, ``single``, ``all``); its manual
examples on the *time* type (``single`` and ``all``) are two more variants. The manual writes
its *time* values as ``2010M1/2010M12``, which vtlengine cannot load: those inputs are rewritten
as ``2010-01-01/2010-12-31`` (adaptations). Its ISO durations such as ``P2Y230D`` are kept:
vtlengine has no equivalent representation, so they are engine findings, not adaptations.

The other operators have the operand levels of their syntax, and ``time_agg`` the optional
parameters (``periodIndFrom``, ``first``/``last``) and its use inside ``group all``. VTL has no
time literals: the scalar date or time period of an example is an input scalar of that type
(``input/scalars.csv``), and the duration of ``monthtoday`` and ``yeartoday`` a ``cast`` of a
string, with no mask (vtlengine implements no cast mask), as a Duration input scalar refuses an
ISO 8601 text.

Every form declares the types it takes and gives, the finest that its page states:

* a time series form takes and gives the type of its reference time Identifier, with the
  ``number`` measures of ``flow_to_stock`` and ``stock_to_flow``, and the ``integer`` shift of
  ``timeshift``. The pages of ``fill_time_series`` and ``timeshift`` leave the measures of their
  Data Sets untyped, and their results keep those of the operand ("the same Identifier, Measure
  and Attribute Components"): their forms give ``any`` too, as the result follows the input;
* ``time_agg`` takes a ``time``, or the ``date`` or ``time_period`` that its form names, and the
  ``duration`` of its period indicators, and gives the type that it takes;
* the pages of ``datediff``, ``dateadd`` and the extractions take "date or time_period" values,
  so their forms take those two types, or the one that the form names. ``dateadd`` gives a
  ``date``, except on a Data Set, whose time Identifier keeps its type and whose measures are
  kept, untyped: that form gives ``any`` too;
* ``period_indicator`` takes a ``time_period`` and gives a ``duration``, and the conversions
  between numbers of days and durations go from ``integer`` to ``duration`` and back;
* ``current_date()`` takes nothing and gives a ``date``.

The forms of ``time_agg`` and ``period_indicator`` whose id has no kind say it: the ``group all``
form ("a component version, for use in combination with an aggregation operator") and the
operand left out in a clause are on components, the other ``time_agg`` forms on scalars.
"""

from __future__ import annotations

from typing import Final

from tool.model import ANY_TYPE, Chapter, Example, Link, Operator, Variant, authored, manual

_TIME_ADAPTATION = (
    "the manual's time values YYYYMm/YYYYMm (e.g. 2010M1/2010M12), which vtlengine cannot load, "
    "are rewritten as YYYY-MM-DD/YYYY-MM-DD (2010-01-01/2010-12-31)"
)
_DUPLICATED_G = (
    "the expected output repeats Id_1 = G on the second data point, where the input has H"
)

_TIME: Final = "time"
_DATE: Final = "date"
_TIME_PERIOD: Final = "time_period"
_DURATION: Final = "duration"
_INTEGER: Final = "integer"
_NUMBER: Final = "number"
# What the pages of datediff, dateadd and the extractions take: "date or time_period" values.
_DATE_OR_PERIOD: Final = (_DATE, _TIME_PERIOD)
# What a form gives besides its reference time Identifier, when its result has the measures of
# its operand and the page does not type them: the result follows the input.
_KEEPS_THE_MEASURES: Final = (ANY_TYPE,)


def _comp_and_sc(
    operator_id: str, takes: tuple[str, ...], gives: str, comp: Example, sc: Example
) -> tuple[Variant, Variant]:
    """The component form inside ``calc`` and the scalar form of an operator of one operand."""
    return (
        Variant("comp", f"calc {operator_id}(comp)", (comp,), takes=takes, gives=(gives,)),
        Variant("sc", f"{operator_id}(sc)", (sc,), takes=takes, gives=(gives,)),
    )


def _extract(operator_id: str, name: str, comp: Example, sc: Example) -> Operator:
    """An operator of the page "Extract time period": a component and a scalar variant.

    Both take "one date or time_period value" and give an integer.
    """
    return Operator(
        id=operator_id,
        name=name,
        symbol=operator_id,
        page="Time operators/Extract time period",
        accepts=("time",),
        returns=("integer",),
        variants=_comp_and_sc(operator_id, _DATE_OR_PERIOD, _INTEGER, comp, sc),
    )


def _days_to_duration(operator_id: str, name: str, comp: Example, sc: Example) -> Operator:
    """An operator of the page "Number days to duration": from an integer to a duration."""
    return Operator(
        id=operator_id,
        name=name,
        symbol=operator_id,
        page="Time operators/Number days to duration",
        accepts=("integer",),
        returns=("duration",),
        variants=_comp_and_sc(operator_id, (_INTEGER,), _DURATION, comp, sc),
    )


def _duration_to_days(operator_id: str, name: str, comp: Example, sc: Example) -> Operator:
    """An operator of the page "Duration to number days": from a duration to an integer."""
    return Operator(
        id=operator_id,
        name=name,
        symbol=operator_id,
        page="Time operators/Duration to number days",
        accepts=("duration",),
        returns=("integer",),
        variants=_comp_and_sc(operator_id, (_DURATION,), _INTEGER, comp, sc),
    )


def _on_time(
    variant_id: str,
    label: str,
    time_type: str,
    examples: tuple[Example, ...],
    takes: tuple[str, ...] = (),
    gives: tuple[str, ...] = (),
) -> Variant:
    """A form of a time series operator on a reference time Identifier of ``time_type``.

    It takes and gives the type of that Identifier, and the ``takes`` and ``gives`` besides it.
    """
    return Variant(
        variant_id,
        label,
        examples,
        takes=(time_type, *takes),
        gives=(time_type, *gives),
    )


def _time_series(
    operator_id: str,
    name: str,
    accepts: tuple[str, ...],
    form: str,
    date: Example,
    time: Example,
    *,
    takes: tuple[str, ...],
    gives: tuple[str, ...],
) -> Operator:
    """``flow_to_stock``, ``stock_to_flow`` or ``timeshift``: one variant per reference time type.

    On each of their pages ``ex_3`` (annual series) and ``ex_4`` (annual and quarterly series)
    are on a *time_period*, ``date`` (``ex_2``) on a *date* and ``time`` (``ex_1``) on a *time*
    reference time Identifier. The page has the name of the operator. Each form takes and gives
    the type of its reference time Identifier, and ``takes`` and ``gives`` besides it.
    """
    return Operator(
        id=operator_id,
        name=name,
        symbol=operator_id,
        page=f"Time operators/{name}",
        accepts=accepts,
        returns=("dataset",),
        variants=tuple(
            _on_time(time_type, f"{form} on {time_type}", time_type, examples, takes, gives)
            for time_type, examples in (
                (_TIME_PERIOD, (manual("ex_3"), manual("ex_4"))),
                (_DATE, (date,)),
                (_TIME, (time,)),
            )
        ),
    )


CHAPTER = Chapter(
    id="time",
    title="Time operators",
    operators=(
        Operator(
            id="period_indicator",
            name="Period indicator",
            symbol="period_indicator",
            page="Time operators/Period indicator",
            accepts=("time_period", "dataset"),
            returns=("duration", "dataset"),
            variants=(
                Variant(
                    "ds",
                    "period_indicator(DS)",
                    (manual("ex_1"),),
                    takes=(_TIME_PERIOD,),
                    gives=(_DURATION,),
                ),
                Variant(
                    "comp",
                    "filter period_indicator(comp)",
                    (manual("ex_2"),),
                    takes=(_TIME_PERIOD,),
                    gives=(_DURATION,),
                ),
                Variant(
                    "sc",
                    "period_indicator(sc)",
                    (authored("quarterly_period"),),
                    takes=(_TIME_PERIOD,),
                    gives=(_DURATION,),
                ),
                # The operand left out in a clause is the time_period Identifier of its Data Set.
                Variant(
                    "in_filter_no_operand",
                    "filter period_indicator()",
                    (authored("keeps_the_quarterly_data_point"),),
                    takes=(_TIME_PERIOD,),
                    gives=(_DURATION,),
                    kind="comp",
                ),
            ),
        ),
        Operator(
            id="fill_time_series",
            name="Fill time series",
            symbol="fill_time_series",
            page="Time operators/Fill time series",
            accepts=("dataset", "time"),
            returns=("dataset",),
            variants=(
                _on_time(
                    "default_tp",
                    "fill_time_series(DS) on time_period",
                    _TIME_PERIOD,
                    (authored("default_is_all"),),
                    gives=_KEEPS_THE_MEASURES,
                ),
                _on_time(
                    "single_tp",
                    "fill_time_series(DS, single) on time_period",
                    _TIME_PERIOD,
                    (manual("ex_5"), manual("ex_7")),
                    gives=_KEEPS_THE_MEASURES,
                ),
                _on_time(
                    "all_tp",
                    "fill_time_series(DS, all) on time_period",
                    _TIME_PERIOD,
                    (manual("ex_6"), manual("ex_8")),
                    gives=_KEEPS_THE_MEASURES,
                ),
                _on_time(
                    "default_date",
                    "fill_time_series(DS) on date",
                    _DATE,
                    (authored("default_is_all"),),
                    gives=_KEEPS_THE_MEASURES,
                ),
                _on_time(
                    "single_date",
                    "fill_time_series(DS, single) on date",
                    _DATE,
                    (manual("ex_3"),),
                    gives=_KEEPS_THE_MEASURES,
                ),
                _on_time(
                    "all_date",
                    "fill_time_series(DS, all) on date",
                    _DATE,
                    (manual("ex_4"),),
                    gives=_KEEPS_THE_MEASURES,
                ),
                _on_time(
                    "single_time",
                    "fill_time_series(DS, single) on time",
                    _TIME,
                    (manual("ex_1", adaptation=_TIME_ADAPTATION),),
                    gives=_KEEPS_THE_MEASURES,
                ),
                _on_time(
                    "all_time",
                    "fill_time_series(DS, all) on time",
                    _TIME,
                    (manual("ex_2", adaptation=_TIME_ADAPTATION),),
                    gives=_KEEPS_THE_MEASURES,
                ),
            ),
        ),
        _time_series(
            "flow_to_stock",
            "Flow to stock",
            ("dataset", "time", "number"),
            "flow_to_stock(DS)",
            manual("ex_2"),
            manual(
                "ex_1",
                adaptation=_TIME_ADAPTATION,
                erratum=(
                    "the expected output declares Me_1 as String; flow_to_stock keeps the "
                    "Integer measure of DS_1"
                ),
            ),
            takes=(_NUMBER,),
            gives=(_NUMBER,),
        ),
        _time_series(
            "stock_to_flow",
            "Stock to flow",
            ("dataset", "time", "number"),
            "stock_to_flow(DS)",
            manual("ex_2"),
            manual(
                "ex_1",
                adaptation=_TIME_ADAPTATION,
                erratum=(
                    "the input DS_1 declares Me_1 as String; stock_to_flow needs a number "
                    "measure, and the values and the result are Numbers"
                ),
            ),
            takes=(_NUMBER,),
            gives=(_NUMBER,),
        ),
        _time_series(
            "timeshift",
            "Time shift",
            ("dataset", "time", "integer"),
            "timeshift(DS, n)",
            manual("ex_2"),
            manual("ex_1", adaptation=_TIME_ADAPTATION),
            takes=(_INTEGER,),
            gives=_KEEPS_THE_MEASURES,
        ),
        Operator(
            id="time_agg",
            name="Time aggregation",
            symbol="time_agg",
            page="Time operators/Time aggregation",
            accepts=("time", "duration", "dataset"),
            returns=("time", "dataset"),
            variants=(
                # The component version "for use in combination with an aggregation operator".
                Variant(
                    "group_all",
                    "sum(DS group all time_agg(p))",
                    (manual("ex_1"),),
                    takes=(_TIME, _DURATION),
                    gives=(_TIME,),
                    kind="comp",
                ),
                Variant(
                    "ds",
                    "time_agg(p, DS)",
                    (authored("monthly_measure_to_quarters"),),
                    takes=(_TIME, _DURATION),
                    gives=(_TIME,),
                ),
                Variant(
                    "comp",
                    "calc time_agg(p, comp)",
                    (authored("mixed_periods_to_years"),),
                    takes=(_TIME, _DURATION),
                    gives=(_TIME,),
                ),
                Variant(
                    "sc_time_period",
                    "time_agg(p, sc) on time_period",
                    (authored("month_to_quarter"),),
                    takes=(_TIME_PERIOD, _DURATION),
                    gives=(_TIME_PERIOD,),
                ),
                Variant(
                    "sc_date",
                    "time_agg(p, sc) on date",
                    (authored("without_first_or_last"),),
                    takes=(_DATE, _DURATION),
                    gives=(_DATE,),
                ),
                Variant(
                    "period_from",
                    "time_agg(p, from, sc)",
                    (authored("quarter_to_year"),),
                    takes=(_TIME, _DURATION),
                    gives=(_TIME,),
                    kind="sc",
                ),
                Variant(
                    "first",
                    "time_agg(p, sc, first) on date",
                    (authored("start_of_the_year"),),
                    takes=(_DATE, _DURATION),
                    gives=(_DATE,),
                    kind="sc",
                ),
                Variant(
                    "last",
                    "time_agg(p, sc, last) on date",
                    (authored("end_of_the_quarter"),),
                    takes=(_DATE, _DURATION),
                    gives=(_DATE,),
                    kind="sc",
                ),
            ),
            links=(
                Link("aggregate.aggregate_invocation", "used in group all"),
                Link("clause.aggr", "used in group all"),
            ),
        ),
        Operator(
            id="current_date",
            name="Actual time",
            symbol="current_date",
            page="Time operators/Actual time",
            accepts=(),
            returns=("date",),
            variants=(
                Variant(
                    "sc",
                    "current_date()",
                    (authored("date_of_the_run", check="current_date"),),
                    takes=(),
                    gives=(_DATE,),
                ),
            ),
        ),
        Operator(
            id="datediff",
            name="Days between two dates",
            symbol="datediff",
            page="Time operators/Days between two dates",
            accepts=("time",),
            returns=("integer",),
            variants=(
                Variant(
                    "comp_comp",
                    "calc datediff(comp, comp)",
                    (manual("ex_1"),),
                    takes=_DATE_OR_PERIOD,
                    gives=(_INTEGER,),
                ),
                Variant(
                    "sc_date",
                    "datediff(date, date)",
                    (authored("across_the_new_year"),),
                    takes=(_DATE,),
                    gives=(_INTEGER,),
                ),
                Variant(
                    "sc_time_period",
                    "datediff(time_period, time_period)",
                    (authored("last_days_of_two_quarters"),),
                    takes=(_TIME_PERIOD,),
                    gives=(_INTEGER,),
                ),
                Variant(
                    "sc_mixed",
                    "datediff(time_period, date)",
                    (authored("quarter_end_to_a_date"),),
                    takes=(_TIME_PERIOD, _DATE),
                    gives=(_INTEGER,),
                ),
            ),
        ),
        Operator(
            id="dateadd",
            name="Add time unit to date",
            symbol="dateadd",
            page="Time operators/Add time unit to date",
            accepts=("time", "integer", "duration", "dataset"),
            returns=("time", "dataset"),
            variants=(
                # A Data Set keeps its components: its time Identifier, and its untyped measures.
                Variant(
                    "ds",
                    "dateadd(DS, n, p)",
                    (authored("shifts_the_time_identifier"),),
                    takes=(_TIME, _INTEGER, _DURATION),
                    gives=(_TIME, *_KEEPS_THE_MEASURES),
                ),
                Variant(
                    "comp",
                    "calc dateadd(comp, n, p)",
                    (manual("ex_1"),),
                    takes=(*_DATE_OR_PERIOD, _INTEGER, _DURATION),
                    gives=(_DATE,),
                ),
                Variant(
                    "sc_date",
                    "dateadd(date, n, p)",
                    (
                        authored("one_month_later"),
                        authored(
                            "one_year_later",
                            adaptation=(
                                "the page writes the year period indicator “Y” (“for years the "
                                "“Y” is used”); the example writes it A, the year duration of "
                                "vtlengine and of the chapter's duration codes"
                            ),
                        ),
                    ),
                    takes=(_DATE, _INTEGER, _DURATION),
                    gives=(_DATE,),
                ),
                Variant(
                    "sc_time_period",
                    "dateadd(time_period, n, p)",
                    (authored("from_the_last_day_of_the_quarter"),),
                    takes=(_TIME_PERIOD, _INTEGER, _DURATION),
                    gives=(_DATE,),
                ),
            ),
        ),
        _extract(
            "getyear",
            "Extract the year",
            authored("dates_of_three_years"),
            authored("quarterly_period"),
        ),
        _extract(
            "getmonth",
            "Extract the month",
            manual("ex_1"),
            authored("monthly_period"),
        ),
        _extract(
            "dayofmonth",
            "Extract the day of the month",
            authored("leap_day_and_month_end"),
            authored("mid_month_date"),
        ),
        _extract(
            "dayofyear",
            "Extract the day of the year",
            authored("leap_and_common_years"),
            authored("april_in_a_leap_year"),
        ),
        _days_to_duration(
            "daytoyear",
            "Number of days to years and days",
            manual("ex_1", erratum=_DUPLICATED_G),
            authored("years_of_365_days"),
        ),
        _days_to_duration(
            "daytomonth",
            "Number of days to months and days",
            manual("ex_2", erratum=_DUPLICATED_G),
            authored("months_of_thirty_days"),
        ),
        _duration_to_days(
            "yeartoday",
            "Years and days to number of days",
            manual(
                "ex_1",
                erratum=(
                    "the expected data points are those of Number days to duration "
                    "ex_1 (Me_1 = 240, Me_2 = P0Y240D...), not yeartoday of DS_1"
                ),
            ),
            authored("one_year_and_twenty_days"),
        ),
        _duration_to_days(
            "monthtoday",
            "Months and days to number of days",
            authored("iso_month_and_day_durations"),
            authored("three_months_and_ten_days"),
        ),
    ),
)
