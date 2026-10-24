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
time literals: a scalar date, time period or duration is written as a ``cast`` of a string, with
no mask (vtlengine implements no cast mask).
"""

from __future__ import annotations

from tool.model import Chapter, Example, Link, Operator, Variant, authored, manual

_TIME_ADAPTATION = (
    "the manual's time values YYYYMm/YYYYMm (e.g. 2010M1/2010M12), which vtlengine cannot load, "
    "are rewritten as YYYY-MM-DD/YYYY-MM-DD (2010-01-01/2010-12-31)"
)
_DUPLICATED_G = (
    "the expected output repeats Id_1 = G on the second data point, where the input has H"
)


def _extract(operator_id: str, name: str, comp: Variant) -> Operator:
    """An operator of the page "Extract time period": a component and a scalar variant."""
    return Operator(
        id=operator_id,
        name=name,
        symbol=operator_id,
        page="Time operators/Extract time period",
        accepts=("time",),
        returns=("integer",),
        variants=(comp, Variant("sc", f"{operator_id}(sc)", (authored(),))),
    )


def _days_to_duration(operator_id: str, name: str, comp: Variant) -> Operator:
    """An operator of the page "Number days to duration"."""
    return Operator(
        id=operator_id,
        name=name,
        symbol=operator_id,
        page="Time operators/Number days to duration",
        accepts=("integer",),
        returns=("duration",),
        variants=(comp, Variant("sc", f"{operator_id}(sc)", (authored(),))),
    )


def _duration_to_days(operator_id: str, name: str, comp: Variant) -> Operator:
    """An operator of the page "Duration to number days"."""
    return Operator(
        id=operator_id,
        name=name,
        symbol=operator_id,
        page="Time operators/Duration to number days",
        accepts=("duration",),
        returns=("integer",),
        variants=(comp, Variant("sc", f"{operator_id}(sc)", (authored(),))),
    )


def _time_series(
    operator_id: str,
    name: str,
    accepts: tuple[str, ...],
    form: str,
    date: Example,
    time: Example,
) -> Operator:
    """``flow_to_stock``, ``stock_to_flow`` or ``timeshift``: one variant per reference time type.

    On each of their pages ``ex_3`` (annual series) and ``ex_4`` (annual and quarterly series)
    are on a *time_period*, ``date`` (``ex_2``) on a *date* and ``time`` (``ex_1``) on a *time*
    reference time Identifier. The page has the name of the operator.
    """
    return Operator(
        id=operator_id,
        name=name,
        symbol=operator_id,
        page=f"Time operators/{name}",
        accepts=accepts,
        returns=("dataset",),
        variants=(
            Variant("time_period", f"{form} on time_period", (manual("ex_3"), manual("ex_4"))),
            Variant("date", f"{form} on date", (date,)),
            Variant("time", f"{form} on time", (time,)),
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
                Variant("ds", "period_indicator(DS)", (manual("ex_1"),)),
                Variant("comp", "filter period_indicator(comp)", (manual("ex_2"),)),
                Variant("sc", "period_indicator(sc)", (authored(),)),
                Variant("in_filter_no_operand", "filter period_indicator()", (authored(),)),
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
                Variant("default_tp", "fill_time_series(DS) on time_period", (authored(),)),
                Variant(
                    "single_tp",
                    "fill_time_series(DS, single) on time_period",
                    (manual("ex_5"), manual("ex_7")),
                ),
                Variant(
                    "all_tp",
                    "fill_time_series(DS, all) on time_period",
                    (manual("ex_6"), manual("ex_8")),
                ),
                Variant("default_date", "fill_time_series(DS) on date", (authored(),)),
                Variant("single_date", "fill_time_series(DS, single) on date", (manual("ex_3"),)),
                Variant("all_date", "fill_time_series(DS, all) on date", (manual("ex_4"),)),
                Variant(
                    "single_time",
                    "fill_time_series(DS, single) on time",
                    (manual("ex_1", adaptation=_TIME_ADAPTATION),),
                ),
                Variant(
                    "all_time",
                    "fill_time_series(DS, all) on time",
                    (manual("ex_2", adaptation=_TIME_ADAPTATION),),
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
        ),
        _time_series(
            "timeshift",
            "Time shift",
            ("dataset", "time", "integer"),
            "timeshift(DS, n)",
            manual("ex_2"),
            manual("ex_1", adaptation=_TIME_ADAPTATION),
        ),
        Operator(
            id="time_agg",
            name="Time aggregation",
            symbol="time_agg",
            page="Time operators/Time aggregation",
            accepts=("time", "duration", "dataset"),
            returns=("time", "dataset"),
            variants=(
                Variant("group_all", "sum(DS group all time_agg(p))", (manual("ex_1"),)),
                Variant("ds", "time_agg(p, DS)", (authored(),)),
                Variant("comp", "calc time_agg(p, comp)", (authored(),)),
                Variant("sc_time_period", "time_agg(p, sc) on time_period", (authored(),)),
                Variant("sc_date", "time_agg(p, sc) on date", (authored(),)),
                Variant("period_from", "time_agg(p, from, sc)", (authored(),)),
                Variant("first", "time_agg(p, sc, first) on date", (authored(),)),
                Variant("last", "time_agg(p, sc, last) on date", (authored(),)),
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
            variants=(Variant("sc", "current_date()", (authored(check="current_date"),)),),
        ),
        Operator(
            id="datediff",
            name="Days between two dates",
            symbol="datediff",
            page="Time operators/Days between two dates",
            accepts=("time",),
            returns=("integer",),
            variants=(
                Variant("comp_comp", "calc datediff(comp, comp)", (manual("ex_1"),)),
                Variant("sc_date", "datediff(date, date)", (authored(),)),
                Variant("sc_time_period", "datediff(time_period, time_period)", (authored(),)),
                Variant("sc_mixed", "datediff(time_period, date)", (authored(),)),
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
                Variant("ds", "dateadd(DS, n, p)", (authored(),)),
                Variant("comp", "calc dateadd(comp, n, p)", (manual("ex_1"),)),
                Variant(
                    "sc_date",
                    "dateadd(date, n, p)",
                    (
                        authored(),
                        authored(
                            "minimal_2",
                            adaptation=(
                                "the page writes the year period indicator “Y” (“for years the "
                                "“Y” is used”); the example writes it A, the year duration of "
                                "vtlengine and of the chapter's duration codes"
                            ),
                        ),
                    ),
                ),
                Variant("sc_time_period", "dateadd(time_period, n, p)", (authored(),)),
            ),
        ),
        _extract(
            "getyear",
            "Extract the year",
            Variant("comp", "calc getyear(comp)", (authored(),)),
        ),
        _extract(
            "getmonth",
            "Extract the month",
            Variant("comp", "calc getmonth(comp)", (manual("ex_1"),)),
        ),
        _extract(
            "dayofmonth",
            "Extract the day of the month",
            Variant("comp", "calc dayofmonth(comp)", (authored(),)),
        ),
        _extract(
            "dayofyear",
            "Extract the day of the year",
            Variant("comp", "calc dayofyear(comp)", (authored(),)),
        ),
        _days_to_duration(
            "daytoyear",
            "Number of days to years and days",
            Variant("comp", "calc daytoyear(comp)", (manual("ex_1", erratum=_DUPLICATED_G),)),
        ),
        _days_to_duration(
            "daytomonth",
            "Number of days to months and days",
            Variant("comp", "calc daytomonth(comp)", (manual("ex_2", erratum=_DUPLICATED_G),)),
        ),
        _duration_to_days(
            "yeartoday",
            "Years and days to number of days",
            Variant(
                "comp",
                "calc yeartoday(comp)",
                (
                    manual(
                        "ex_1",
                        erratum=(
                            "the expected data points are those of Number days to duration "
                            "ex_1 (Me_1 = 240, Me_2 = P0Y240D...), not yeartoday of DS_1"
                        ),
                    ),
                ),
            ),
        ),
        _duration_to_days(
            "monthtoday",
            "Months and days to number of days",
            Variant("comp", "calc monthtoday(comp)", (authored(),)),
        ),
    ),
)
