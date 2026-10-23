"""Aggregate and analytic operators: functions over groups and over windows of data points.

The ten aggregate functions (``avg``, ``count``, ``max``, ``median``, ``min``, ``stddev_pop``,
``stddev_samp``, ``sum``, ``var_pop``, ``var_samp``) are invoked at data set level with an
optional grouping clause (``group by``, ``group except``, the deprecated ``group all`` with a
conversion such as ``time_agg``, and ``having``), at component level inside an ``aggr`` clause,
or as analytic functions with ``over ( ... )``. ``first_value``, ``last_value``, ``lag``,
``lead``, ``rank`` and ``ratio_to_report`` are analytic only. The two invocation pages describe
the grouping clause and the analytic clause (partition, order and window) once for all of them.

In the Aggregate invocation, the data set form of ``count`` returns a single Integer measure
``int_var``, whatever the measures of the operand, and ignores the data points whose measures are
all null: that is the team decision of #1049, so those examples carry ``decision=1049``. The
decision does not cover the analytic invocation, whose data set form follows the single result
type of the count page without a decision. Every ``having`` example references only aggregated
expressions of the aggregated measures, where the manual and the reading of #1119 agree, so none
of them needs ``decision=1119``.

The authored examples share one input, ``DS_1`` with a String and a Time_Period identifier and
two Number measures, and no null. The exceptions are ``minimal_2`` examples: those of
``ds_having`` drop the second measure; that of ``count.ds_group_by`` has a data point whose
measures are all null and one with a single null measure; that of ``avg.ds_group_by`` has one
null value; and those of ``min.ds_group_by`` and ``max.ds_group_by`` have a String measure. The
notes of each example derive its expected output.
"""

from __future__ import annotations

from tool.model import Chapter, Example, Operator, Variant, authored, manual

_CHAPTER_TITLE = "Aggregate and Analytic operators"
_COUNT_DECISION = 1049  # count over a data set returns the single measure int_var
# The functions whose ``group by`` form has a second example, ``minimal_2``: nulls for ``count``
# and ``avg``, a String measure for ``min`` and ``max``.
_GROUP_BY_SECOND = ("count", "avg", "min", "max")


def _page(title: str) -> str:
    return f"{_CHAPTER_TITLE}/{title}"


def _aggregate_function(op_id: str, name: str, title: str, accepts: str, returns: str) -> Operator:
    """One of the ten aggregate functions, with its eight forms (nine for ``count``).

    Each function has the manual example ``ex_1``, a ``group by`` over one identifier; ``count``
    also has ``ex_2``, ``count ( )`` in a ``having`` condition.
    """
    is_count = op_id == "count"
    decision = _COUNT_DECISION if is_count else None

    def dataset_form() -> tuple[Example, ...]:
        return (authored(decision=decision),)

    # ``minimal`` has the two measures of the shared input; ``minimal_2`` the first one only.
    having: tuple[Variant, ...] = (
        Variant(
            "ds_having",
            f"{op_id}(DS group by Id having cond)",
            (authored(decision=decision), authored("minimal_2", decision=decision)),
        ),
    )
    if is_count:
        having += (
            Variant(
                "count_in_having",
                "count(DS group by Id having count() > n)",
                (manual("ex_2", decision=decision),),
            ),
        )
    group_by: tuple[Example, ...] = (manual("ex_1", decision=decision),)
    if op_id in _GROUP_BY_SECOND:
        group_by += (authored("minimal_2", decision=decision),)
    variants = (
        Variant("ds_no_group", f"{op_id}(DS)", dataset_form()),
        Variant("ds_group_by", f"{op_id}(DS group by Id)", group_by),
        Variant("ds_group_except", f"{op_id}(DS group except Id)", dataset_form()),
        Variant("ds_group_all", f'{op_id}(DS group all time_agg("A"))', dataset_form()),
        *having,
        Variant("comp_in_aggr", f"DS[aggr Me := {op_id}(Me) group by Id]", (authored(),)),
        # #1049 covers the Aggregate invocation only: no decision on the analytic form.
        Variant("analytic_ds", f"{op_id}(DS over (partition by Id))", (authored(),)),
        Variant(
            "analytic_comp",
            f"DS[calc Me := {op_id}(Me over (partition by Id))]",
            (authored(),),
        ),
    )
    return Operator(
        id=op_id,
        name=name,
        symbol=op_id,
        page=_page(title),
        accepts=(accepts,),
        returns=(returns,),
        variants=variants,
    )


def _first_or_last(op_id: str, name: str) -> Operator:
    return Operator(
        id=op_id,
        name=name,
        symbol=op_id,
        page=_page(name),
        accepts=("scalar",),
        returns=("scalar",),
        variants=(
            Variant(
                "analytic_ds",
                f"{op_id}(DS over (partition by Id order by Id data points ...))",
                (manual("ex_1"),),
            ),
            Variant(
                "analytic_comp",
                f"DS[calc Me := {op_id}(Me over (partition by Id order by Id data points ...))]",
                (authored(),),
            ),
        ),
    )


def _lag_or_lead(op_id: str, name: str) -> Operator:
    return Operator(
        id=op_id,
        name=name,
        symbol=op_id,
        page=_page(name),
        accepts=("scalar",),
        returns=("scalar",),
        variants=(
            Variant(
                "ds_no_offset", f"{op_id}(DS over (partition by Id order by Id))", (authored(),)
            ),
            Variant(
                "ds_offset",
                f"{op_id}(DS, n over (partition by Id order by Id))",
                (manual("ex_1"),),
            ),
            Variant(
                "ds_offset_default",
                f"{op_id}(DS, n, default over (partition by Id order by Id))",
                (authored(),),
            ),
            Variant(
                "comp_offset",
                f"DS[calc Me := {op_id}(Me, n over (partition by Id order by Id))]",
                (authored(),),
            ),
        ),
    )


_AGGREGATE_INVOCATION = Operator(
    id="aggregate_invocation",
    name="Aggregate invocation",
    symbol="group by",
    page=_page("Aggregate invocation"),
    accepts=("dataset",),
    returns=("dataset",),
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
        ),
        Variant("group_by_two", "agg(DS group by Id, Id)", (manual("ex_2"),)),
        Variant("no_group", "agg(DS)", (manual("ex_3"),)),
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
        ),
    ),
)

_ANALYTIC_INVOCATION = Operator(
    id="analytic_invocation",
    name="Analytic invocation",
    symbol="over",
    page=_page("Analytic invocation"),
    accepts=("dataset",),
    returns=("dataset",),
    variants=(
        Variant("partition_by", "an(Me over (partition by Id))", (authored(),)),
        Variant("partition_except", "an(Me over (partition except Id))", (authored(),)),
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
                authored(),
            ),
        ),
        Variant("order_asc", "an(over (partition by Id order by Me asc))", (authored(),)),
        Variant("order_desc", "an(over (partition by Id order by Me desc))", (authored(),)),
        Variant(
            "window_data_points",
            "an(DS over (order by Id data points between n preceding and n following))",
            (manual("ex_1"),),
        ),
        Variant(
            "window_range",
            "an(Me over (partition by Id order by Me range between n preceding and n following))",
            (authored(),),
        ),
        Variant(
            "window_unbounded",
            "an(Me over (... data points between unbounded preceding and unbounded following))",
            (authored(),),
        ),
        Variant(
            "window_current",
            "an(Me over (... data points between n preceding and current data point))",
            (authored(),),
        ),
    ),
)

_RANK = Operator(
    id="rank",
    name="Rank",
    symbol="rank",
    page=_page("Rank"),
    accepts=("scalar",),
    returns=("integer",),
    variants=(
        Variant(
            "partition_order",
            "DS[calc Me := rank(over (partition by Id order by Me))]",
            (manual("ex_1"),),
        ),
        Variant("order_only", "DS[calc Me := rank(over (order by Id))]", (authored(),)),
    ),
)

_RATIO_TO_REPORT = Operator(
    id="ratio_to_report",
    name="Ratio to report",
    symbol="ratio_to_report",
    page=_page("Ratio to report"),
    accepts=("number",),
    returns=("number",),
    variants=(
        Variant("ds", "ratio_to_report(DS over (partition by Id))", (manual("ex_1"),)),
        Variant(
            "comp",
            "DS[calc Me := ratio_to_report(Me over (partition by Id))]",
            (authored(),),
        ),
    ),
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
            "dataset",
            "integer",
        ),
        _aggregate_function("min", "Minimum value", "Minimun value", "scalar", "scalar"),
        _aggregate_function("max", "Maximum value", "Maximum value", "scalar", "scalar"),
        _aggregate_function("median", "Median value", "Median value", "number", "number"),
        _aggregate_function("sum", "Sum", "Sum", "number", "number"),
        _aggregate_function("avg", "Average value", "Average value", "number", "number"),
        _aggregate_function(
            "stddev_pop",
            "Population standard deviation",
            "Population standard deviation",
            "number",
            "number",
        ),
        _aggregate_function(
            "stddev_samp",
            "Sample standard deviation",
            "Sample standard deviation",
            "number",
            "number",
        ),
        _aggregate_function(
            "var_pop", "Population variance", "Population variance", "number", "number"
        ),
        _aggregate_function("var_samp", "Sample variance", "Sample variance", "number", "number"),
        _first_or_last("first_value", "First value"),
        _first_or_last("last_value", "Last value"),
        _lag_or_lead("lag", "Lag"),
        _lag_or_lead("lead", "Lead"),
        _RANK,
        _RATIO_TO_REPORT,
    ),
)
