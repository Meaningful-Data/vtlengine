"""Join operators: ``inner_join``, ``left_join``, ``full_join`` and ``cross_join``.

The four operators take two or more data sets and an optional body of clauses: ``filter``,
then one of ``apply``, ``calc`` (with a role) or ``aggr`` (with a grouping and ``having``), then
``keep`` or ``drop``, then ``rename``. ``left_join`` and ``full_join`` also take ``using`` with
``nvl(...)`` rules for the identifiers that are not join keys; ``cross_join`` takes no ``using``.

Each operator has one variant per form (join kind x clause), with the other parts of the body
left out. Every variant has an authored ``minimal`` example on the same inputs for the four
kinds, so the kinds can be compared form by form; the manual's examples are added to the variant
whose clause they exercise.
"""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

CHAPTER = Chapter(
    id="join",
    title="Join operators",
    operators=(
        Operator(
            id="inner_join",
            name="Inner Join",
            symbol="inner_join",
            page="Join operators/Inner Join",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                Variant(id="bare", label="inner_join(DS, DS)", examples=(authored(),)),
                Variant(id="aliases", label="inner_join(DS as a, DS as b)", examples=(authored(),)),
                Variant(id="using", label="inner_join(DS, DS using Id)", examples=(authored(),)),
                Variant(
                    id="filter", label="inner_join(DS, DS filter cond)", examples=(authored(),)
                ),
                Variant(
                    id="apply",
                    label="inner_join(DS, DS apply expr)",
                    examples=(manual("ex_4"), authored()),
                ),
                Variant(
                    id="calc_measure",
                    label="inner_join(DS, DS calc measure C := expr)",
                    examples=(manual("ex_3"), authored()),
                ),
                Variant(
                    id="calc_identifier",
                    label="inner_join(DS, DS calc identifier C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="calc_attribute",
                    label="inner_join(DS, DS calc attribute C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="calc_viral_attribute",
                    label="inner_join(DS, DS calc viral attribute C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_group_by",
                    label="inner_join(DS, DS aggr C := agg group by Id)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_group_except",
                    label="inner_join(DS, DS aggr C := agg group except Id)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_having",
                    label="inner_join(DS, DS aggr C := agg group by Id having cond)",
                    examples=(authored(),),
                ),
                Variant(
                    id="keep",
                    label="inner_join(DS, DS keep C)",
                    examples=(manual("ex_1"), authored()),
                ),
                Variant(
                    id="drop",
                    label="inner_join(DS, DS drop C)",
                    examples=(manual("ex_2"), authored()),
                ),
                Variant(
                    id="rename", label="inner_join(DS, DS rename C to C2)", examples=(authored(),)
                ),
                Variant(
                    id="nary",
                    label="inner_join(DS, DS, DS)",
                    examples=(
                        manual(
                            "ex_5",
                            erratum=(
                                "Me_3 is in both IBSC and IBSD, so stripping the alias prefixes "
                                "gives two Me_3, an error by the Behaviour section; the script "
                                "adds drop IBSC#Me_3 (all null), which gives the printed result. "
                                "ds_5.json also declares Me_2 an Identifier, but the result has "
                                "Me_21 and Me_22 as Measures: it is loaded as a Measure."
                            ),
                        ),
                        authored(),
                    ),
                ),
            ),
        ),
        Operator(
            id="left_join",
            name="Left Join",
            symbol="left_join",
            page="Join operators/Left Join",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                Variant(id="bare", label="left_join(DS, DS)", examples=(authored(),)),
                Variant(id="aliases", label="left_join(DS as a, DS as b)", examples=(authored(),)),
                Variant(id="using", label="left_join(DS, DS using Id)", examples=(authored(),)),
                Variant(
                    id="using_nvl",
                    label="left_join(DS, DS using Id, nvl(Id, c))",
                    examples=(authored(),),
                ),
                Variant(id="filter", label="left_join(DS, DS filter cond)", examples=(authored(),)),
                Variant(id="apply", label="left_join(DS, DS apply expr)", examples=(authored(),)),
                Variant(
                    id="calc_measure",
                    label="left_join(DS, DS calc measure C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="calc_identifier",
                    label="left_join(DS, DS calc identifier C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="calc_attribute",
                    label="left_join(DS, DS calc attribute C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="calc_viral_attribute",
                    label="left_join(DS, DS calc viral attribute C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_group_by",
                    label="left_join(DS, DS aggr C := agg group by Id)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_group_except",
                    label="left_join(DS, DS aggr C := agg group except Id)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_having",
                    label="left_join(DS, DS aggr C := agg group by Id having cond)",
                    examples=(authored(),),
                ),
                Variant(
                    id="keep",
                    label="left_join(DS, DS keep C)",
                    examples=(manual("ex_1"), authored()),
                ),
                Variant(id="drop", label="left_join(DS, DS drop C)", examples=(authored(),)),
                Variant(
                    id="rename", label="left_join(DS, DS rename C to C2)", examples=(authored(),)
                ),
                Variant(id="nary", label="left_join(DS, DS, DS)", examples=(authored(),)),
            ),
        ),
        Operator(
            id="full_join",
            name="Full Join",
            symbol="full_join",
            page="Join operators/Full Join",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                Variant(id="bare", label="full_join(DS, DS)", examples=(authored(),)),
                Variant(id="aliases", label="full_join(DS as a, DS as b)", examples=(authored(),)),
                Variant(id="using", label="full_join(DS, DS using Id)", examples=(authored(),)),
                Variant(
                    id="using_nvl",
                    label="full_join(DS, DS using Id, nvl(Id, c))",
                    examples=(authored(),),
                ),
                Variant(id="filter", label="full_join(DS, DS filter cond)", examples=(authored(),)),
                Variant(id="apply", label="full_join(DS, DS apply expr)", examples=(authored(),)),
                Variant(
                    id="calc_measure",
                    label="full_join(DS, DS calc measure C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="calc_identifier",
                    label="full_join(DS, DS calc identifier C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="calc_attribute",
                    label="full_join(DS, DS calc attribute C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="calc_viral_attribute",
                    label="full_join(DS, DS calc viral attribute C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_group_by",
                    label="full_join(DS, DS aggr C := agg group by Id)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_group_except",
                    label="full_join(DS, DS aggr C := agg group except Id)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_having",
                    label="full_join(DS, DS aggr C := agg group by Id having cond)",
                    examples=(authored(),),
                ),
                Variant(
                    id="keep",
                    label="full_join(DS, DS keep C)",
                    examples=(manual("ex_1"), authored()),
                ),
                Variant(id="drop", label="full_join(DS, DS drop C)", examples=(authored(),)),
                Variant(
                    id="rename", label="full_join(DS, DS rename C to C2)", examples=(authored(),)
                ),
                Variant(id="nary", label="full_join(DS, DS, DS)", examples=(authored(),)),
            ),
        ),
        Operator(
            id="cross_join",
            name="Cross Join",
            symbol="cross_join",
            page="Join operators/Cross Join",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                Variant(id="bare", label="cross_join(DS, DS)", examples=(authored(),)),
                Variant(id="aliases", label="cross_join(DS as a, DS as b)", examples=(authored(),)),
                Variant(
                    id="filter", label="cross_join(DS, DS filter cond)", examples=(authored(),)
                ),
                Variant(id="apply", label="cross_join(DS, DS apply expr)", examples=(authored(),)),
                Variant(
                    id="calc_measure",
                    label="cross_join(DS, DS calc measure C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="calc_identifier",
                    label="cross_join(DS, DS calc identifier C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="calc_attribute",
                    label="cross_join(DS, DS calc attribute C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="calc_viral_attribute",
                    label="cross_join(DS, DS calc viral attribute C := expr)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_group_by",
                    label="cross_join(DS, DS aggr C := agg group by Id)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_group_except",
                    label="cross_join(DS, DS aggr C := agg group except Id)",
                    examples=(authored(),),
                ),
                Variant(
                    id="aggr_having",
                    label="cross_join(DS, DS aggr C := agg group by Id having cond)",
                    examples=(authored(),),
                ),
                Variant(id="keep", label="cross_join(DS, DS keep C)", examples=(authored(),)),
                Variant(id="drop", label="cross_join(DS, DS drop C)", examples=(authored(),)),
                Variant(
                    id="rename",
                    label="cross_join(DS, DS rename C to C2)",
                    examples=(manual("ex_1"), authored()),
                ),
                Variant(id="nary", label="cross_join(DS, DS, DS)", examples=(authored(),)),
            ),
        ),
    ),
)
