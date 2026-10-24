"""Aggregate and analytic operators: functions over groups and over windows of data points.

The ten aggregate functions (``avg``, ``count``, ``max``, ``median``, ``min``, ``stddev_pop``,
``stddev_samp``, ``sum``, ``var_pop``, ``var_samp``) are invoked at data set level with an
optional grouping clause (``group by``, ``group except``, the deprecated ``group all`` with a
conversion such as ``time_agg``, and ``having``), at component level inside an ``aggr`` clause,
or as analytic functions with ``over ( ... )``. ``first_value``, ``last_value``, ``lag``,
``lead``, ``rank`` and ``ratio_to_report`` are analytic only. The two invocation pages describe
the grouping clause and the analytic clause (partition, order and window) once for all of them.

The types of the forms come from the "Input parameters type" and the "Result type" of each page,
and from the table at the start of the chapter. ``count``, ``min``, ``max``, ``first_value``,
``last_value``, ``lag``, ``lead`` and ``rank`` take any scalar; the other functions take numbers.
``count`` and ``rank`` give integers; ``min``, ``max``, ``first_value``, ``last_value``, ``lag``
and ``lead`` give the type of their operand (``any``); the others give numbers, ``sum`` included,
whose result the manual declares a number. A form with ``having`` takes the boolean of its
condition too, and a form of ``lag`` or ``lead`` with an offset takes the integer of the offset.
The two invocation pages have no type of their own: their forms take any scalar and give the
type of the function that they invoke.

In the Aggregate invocation, the data set form of ``count`` returns a single Integer measure
``int_var``, whatever the measures of the operand, and ignores the data points whose measures are
all null: that is the team decision of #1049. ``decision=1049`` marks only the examples whose
output follows it beyond the manual: an operand with two measures, where the decision settles the
manual's conflict between the count page's ``int_var`` and the Aggregate invocation's "same
Measures as the operand", and data points with null measures. The single-measure examples without
nulls give the manual's own ``int_var`` and carry no mark, and neither does the analytic
invocation, which the decision does not cover. Every ``having`` example references only aggregated
expressions of the aggregated measures, where the manual and the reading of #1119 agree, so none
of them needs ``decision=1119``.

The authored examples share one input, ``DS_1`` with a String and a Time_Period identifier and
two Number measures, and no null. The exceptions are the ``one_measure`` examples of
``ds_having``, which drop the second measure, and the second examples of ``ds_group_by``: that of
``count`` has a data point whose measures are all null and one with a single null measure, that
of ``avg`` has one null value, and those of ``min`` and ``max`` have a String measure. The notes
of each example say what it tests and why it gives its expected output.
"""

from __future__ import annotations

from tool.model import (
    ANY_SCALAR,
    ANY_TYPE,
    Chapter,
    Example,
    Link,
    Operator,
    Variant,
    authored,
    manual,
)

_CHAPTER_TITLE = "Aggregate and Analytic operators"
_COUNT_DECISION = 1049  # count over a data set returns the single measure int_var
_NUMBER = "number"
_INTEGER = "integer"  # what count and rank give, and the offset of lag and lead
_BOOLEAN = "boolean"  # the condition of having
# The functions whose ``group by`` form has a second example, after the manual's: nulls for
# ``count`` and ``avg``, a String measure for ``min`` and ``max``.
_GROUP_BY_SECOND = {
    "count": "all_null_data_point_ignored",
    "avg": "null_left_out_of_average",
    "min": "string_measure",
    "max": "string_measure",
}
# The clauses that the functions are used in: an aggregate function in ``aggr``, and any function
# that takes ``over ( ... )``, the aggregate ones included, in ``calc``.
_AGGR_LINK = Link("clause.aggr", "used in")
_OVER_LINK = Link("clause.calc", "used in over()")


def _page(title: str) -> str:
    return f"{_CHAPTER_TITLE}/{title}"


def _with_dataset(type_id: str) -> tuple[str, ...]:
    """``type_id`` and ``dataset``: a function with a data set form takes and gives one too."""
    return (type_id, "dataset")


def _aggregate_function(op_id: str, name: str, title: str, takes: str, gives: str) -> Operator:
    """One of the ten aggregate functions, with its eight forms (nine for ``count``).

    ``takes`` is the type of its operand and ``gives`` the type of its result; a form with
    ``having`` takes the boolean of its condition too. Each function has the manual example
    ``ex_1``, a ``group by`` over one identifier; ``count`` also has ``ex_2``, ``count ( )`` in a
    ``having`` condition.
    """
    is_count = op_id == "count"
    decision = _COUNT_DECISION if is_count else None
    operand = (takes,)
    with_condition = (takes, _BOOLEAN)
    result = (gives,)
    # ``two_measures`` has the two measures of the shared input; ``one_measure`` the first one
    # only, which gives the manual's own ``int_var`` for ``count``: no decision there.
    having: tuple[Variant, ...] = (
        Variant(
            "ds_having",
            f"{op_id}(DS group by Id having cond)",
            (authored("two_measures", decision=decision), authored("one_measure")),
            takes=with_condition,
            gives=result,
        ),
    )
    if is_count:
        having += (
            Variant(
                "count_in_having",
                "count(DS group by Id having count() > n)",
                (manual("ex_2"),),
                takes=with_condition,
                gives=result,
                kind="ds",
            ),
        )
    # ex_1 has a single measure and no null: the manual's own ``int_var``, no decision.
    group_by: tuple[Example, ...] = (manual("ex_1"),)
    if op_id in _GROUP_BY_SECOND:
        group_by += (authored(_GROUP_BY_SECOND[op_id], decision=decision),)
    variants = (
        Variant(
            "ds_no_group",
            f"{op_id}(DS)",
            (authored("single_group_no_identifiers", decision=decision),),
            takes=operand,
            gives=result,
        ),
        Variant("ds_group_by", f"{op_id}(DS group by Id)", group_by, takes=operand, gives=result),
        Variant(
            "ds_group_except",
            f"{op_id}(DS group except Id)",
            (authored("drops_the_listed_identifier", decision=decision),),
            takes=operand,
            gives=result,
        ),
        Variant(
            "ds_group_all",
            f'{op_id}(DS group all time_agg("A"))',
            (authored("quarters_to_years", decision=decision),),
            takes=operand,
            gives=result,
        ),
        *having,
        Variant(
            "comp_in_aggr",
            f"DS[aggr Me := {op_id}(Me) group by Id]",
            (authored("keeps_only_calculated_measure"),),
            takes=operand,
            gives=result,
        ),
        # #1049 covers the Aggregate invocation only: no decision on the analytic form.
        Variant(
            "analytic_ds",
            f"{op_id}(DS over (partition by Id))",
            (authored("whole_partition_without_order"),),
            takes=operand,
            gives=result,
        ),
        Variant(
            "analytic_comp",
            f"DS[calc Me := {op_id}(Me over (partition by Id))]",
            (authored("whole_partition_without_order"),),
            takes=operand,
            gives=result,
        ),
    )
    return Operator(
        id=op_id,
        name=name,
        symbol=op_id,
        page=_page(title),
        accepts=_with_dataset(takes),
        returns=_with_dataset(gives),
        variants=variants,
        links=(_AGGR_LINK, _OVER_LINK),
    )


def _first_or_last(op_id: str, name: str, slug: str) -> Operator:
    """``first_value`` or ``last_value``: any scalar, and a result of its type."""
    return Operator(
        id=op_id,
        name=name,
        symbol=op_id,
        page=_page(name),
        accepts=_with_dataset(ANY_SCALAR),
        returns=_with_dataset(ANY_TYPE),
        variants=(
            Variant(
                "analytic_ds",
                f"{op_id}(DS over (partition by Id order by Id data points ...))",
                (manual("ex_1"),),
                takes=(ANY_SCALAR,),
                gives=(ANY_TYPE,),
            ),
            Variant(
                "analytic_comp",
                f"DS[calc Me := {op_id}(Me over (partition by Id order by Id data points ...))]",
                (authored(slug),),
                takes=(ANY_SCALAR,),
                gives=(ANY_TYPE,),
            ),
        ),
        links=(_OVER_LINK,),
    )


def _lag_or_lead(op_id: str, name: str, comp_slug: str) -> Operator:
    """``lag`` or ``lead``: any scalar, an integer offset, and a result of the operand's type."""
    with_offset = (ANY_SCALAR, _INTEGER)
    return Operator(
        id=op_id,
        name=name,
        symbol=op_id,
        page=_page(name),
        accepts=(ANY_SCALAR, _INTEGER, "dataset"),
        returns=_with_dataset(ANY_TYPE),
        variants=(
            Variant(
                "ds_no_offset",
                f"{op_id}(DS over (partition by Id order by Id))",
                (authored("offset_defaults_to_one"),),
                takes=(ANY_SCALAR,),
                gives=(ANY_TYPE,),
            ),
            Variant(
                "ds_offset",
                f"{op_id}(DS, n over (partition by Id order by Id))",
                (manual("ex_1"),),
                takes=with_offset,
                gives=(ANY_TYPE,),
            ),
            Variant(
                "ds_offset_default",
                f"{op_id}(DS, n, default over (partition by Id order by Id))",
                (authored("offset_two_default_zero"),),
                takes=with_offset,
                gives=(ANY_TYPE,),
            ),
            Variant(
                "comp_offset",
                f"DS[calc Me := {op_id}(Me, n over (partition by Id order by Id))]",
                (authored(comp_slug),),
                takes=with_offset,
                gives=(ANY_TYPE,),
            ),
        ),
        links=(_OVER_LINK,),
    )


# The invocation pages have no type of their own: their forms take any scalar, and give the type
# of the function that they invoke.
_INVOCATION_TAKES = (ANY_SCALAR,)
_INVOCATION_GIVES = (ANY_TYPE,)

_AGGREGATE_INVOCATION = Operator(
    id="aggregate_invocation",
    name="Aggregate invocation",
    symbol="group by",
    page=_page("Aggregate invocation"),
    accepts=_with_dataset(ANY_SCALAR),
    returns=_with_dataset(ANY_TYPE),
    variants=(
        Variant(
            "group_by",
            "agg(DS group by Id)",
            (
                manual(
                    "ex_1",
                    erratum=(
                        "The script comments its alternative forms with '#', which is not a VTL "
                        "comment (the manual's conventions define /* */ and //): they are "
                        "commented with // here."
                    ),
                ),
            ),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
        Variant(
            "group_by_two",
            "agg(DS group by Id, Id)",
            (manual("ex_2"),),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
        Variant(
            "no_group",
            "agg(DS)",
            (manual("ex_3"),),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
        Variant(
            "aggr_multiple",
            "DS[aggr Me := agg(Me), Me := agg(Me) group by Id]",
            (
                manual(
                    "ex_4",
                    erratum=(
                        "The result is named DS_3, not DS_r, and keeps At_1 as a ViralAttribute; "
                        "DS_1 declares At_1 a plain (non-viral) Attribute, which the default "
                        "propagation rule drops, so DS_r has Id_1, Me_2 and Me_3 only."
                    ),
                ),
            ),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
    ),
    links=(_AGGR_LINK,),
)

_ANALYTIC_INVOCATION = Operator(
    id="analytic_invocation",
    name="Analytic invocation",
    symbol="over",
    page=_page("Analytic invocation"),
    accepts=_with_dataset(ANY_SCALAR),
    returns=_with_dataset(ANY_TYPE),
    variants=(
        Variant(
            "partition_by",
            "an(Me over (partition by Id))",
            (authored("sum_per_quarter"),),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
        Variant(
            "partition_except",
            "an(Me over (partition except Id))",
            (authored("partition_by_unlisted_identifier"),),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
        Variant(
            "partition_except_all",
            "an(over (partition except all order by Id))",
            (
                manual(
                    "ex_2",
                    erratum=(
                        "The result declares Me_1 and Me_2 Number: calc leaves Me_1 Integer as in "
                        "DS_1, and rank returns integer, so both are Integer."
                    ),
                ),
                authored("whole_data_set_one_partition"),
            ),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
        Variant(
            "order_asc",
            "an(over (partition by Id order by Me asc))",
            (authored("ascending_rank_with_tie"),),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
        Variant(
            "order_desc",
            "an(over (partition by Id order by Me desc))",
            (authored("descending_rank_with_tie"),),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
        Variant(
            "window_data_points",
            "an(DS over (order by Id data points between n preceding and n following))",
            (manual("ex_1"),),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
        Variant(
            "window_range",
            "an(Me over (partition by Id order by Me range between n preceding and n following))",
            (authored("range_bounds_are_inclusive"),),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
        Variant(
            "window_unbounded",
            "an(Me over (... data points between unbounded preceding and unbounded following))",
            (authored("unbounded_covers_whole_partition"),),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
        Variant(
            "window_current",
            "an(Me over (... data points between n preceding and current data point))",
            (authored("one_preceding_to_current"),),
            takes=_INVOCATION_TAKES,
            gives=_INVOCATION_GIVES,
        ),
    ),
    links=(Link("clause.calc", "used in"),),
)

# rank has no operand: it ranks by the components of its order clause, of any type.
_RANK = Operator(
    id="rank",
    name="Rank",
    symbol="rank",
    page=_page("Rank"),
    accepts=(ANY_SCALAR,),
    returns=(_INTEGER,),
    variants=(
        Variant(
            "partition_order",
            "DS[calc Me := rank(over (partition by Id order by Me))]",
            (manual("ex_1"),),
            takes=(ANY_SCALAR,),
            gives=(_INTEGER,),
        ),
        Variant(
            "order_only",
            "DS[calc Me := rank(over (order by Id))]",
            (authored("default_partition_by_other_identifiers"),),
            takes=(ANY_SCALAR,),
            gives=(_INTEGER,),
        ),
    ),
    links=(_OVER_LINK,),
)

_RATIO_TO_REPORT = Operator(
    id="ratio_to_report",
    name="Ratio to report",
    symbol="ratio_to_report",
    page=_page("Ratio to report"),
    accepts=_with_dataset(_NUMBER),
    returns=_with_dataset(_NUMBER),
    variants=(
        Variant(
            "ds",
            "ratio_to_report(DS over (partition by Id))",
            (manual("ex_1"),),
            takes=(_NUMBER,),
            gives=(_NUMBER,),
        ),
        Variant(
            "comp",
            "DS[calc Me := ratio_to_report(Me over (partition by Id))]",
            (authored("share_of_partition_sum"),),
            takes=(_NUMBER,),
            gives=(_NUMBER,),
        ),
    ),
    links=(_OVER_LINK,),
)

CHAPTER = Chapter(
    id="aggregate",
    title=_CHAPTER_TITLE,
    operators=(
        _AGGREGATE_INVOCATION,
        _ANALYTIC_INVOCATION,
        _aggregate_function(
            "count",
            "Counting the number of data points",
            "Counting the number of data points",
            ANY_SCALAR,
            _INTEGER,
        ),
        _aggregate_function("min", "Minimum value", "Minimun value", ANY_SCALAR, ANY_TYPE),
        _aggregate_function("max", "Maximum value", "Maximum value", ANY_SCALAR, ANY_TYPE),
        _aggregate_function("median", "Median value", "Median value", _NUMBER, _NUMBER),
        _aggregate_function("sum", "Sum", "Sum", _NUMBER, _NUMBER),
        _aggregate_function("avg", "Average value", "Average value", _NUMBER, _NUMBER),
        _aggregate_function(
            "stddev_pop",
            "Population standard deviation",
            "Population standard deviation",
            _NUMBER,
            _NUMBER,
        ),
        _aggregate_function(
            "stddev_samp",
            "Sample standard deviation",
            "Sample standard deviation",
            _NUMBER,
            _NUMBER,
        ),
        _aggregate_function(
            "var_pop", "Population variance", "Population variance", _NUMBER, _NUMBER
        ),
        _aggregate_function("var_samp", "Sample variance", "Sample variance", _NUMBER, _NUMBER),
        _first_or_last("first_value", "First value", "first_in_order_not_smallest"),
        _first_or_last("last_value", "Last value", "last_in_order_not_greatest"),
        _lag_or_lead("lag", "Lag", "previous_value_or_null"),
        _lag_or_lead("lead", "Lead", "next_value_or_null"),
        _RANK,
        _RATIO_TO_REPORT,
    ),
)
