"""Clause operators: the clauses written in square brackets after a data set, ``DS[...]``.

Each clause takes one data set and returns a data set. ``calc``, ``aggr`` and ``filter`` hold
expressions at component level; ``keep``, ``drop``, ``rename``, ``pivot`` and ``unpivot`` name
components; ``sub`` fixes identifiers to scalar values. None of them applies to scalars.

``calc identifier`` over an expression declared nullable raises 1-1-1-16 by the team decision
of #1010, so the ``identifier`` example declares its expression non-nullable.
"""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

CHAPTER = Chapter(
    id="clause",
    title="Clause operators",
    operators=(
        Operator(
            id="calc",
            name="Calculation of a Component",
            symbol="calc",
            page="Clause operators/Calculation of a Component",
            accepts=("dataset", "scalar"),
            returns=("dataset",),
            variants=(
                Variant(
                    "measure_default",
                    "DS[calc Me := expr]",
                    (manual("ex_1"), authored(), authored("minimal_2")),
                ),
                Variant(
                    "measure_explicit",
                    "DS[calc measure Me := expr]",
                    (authored(), authored("minimal_2")),
                ),
                Variant(
                    "identifier",
                    "DS[calc identifier Id := expr]",
                    (authored(decision=1010),),
                ),
                Variant("attribute", "DS[calc attribute At := expr]", (manual("ex_2"),)),
                Variant(
                    "viral_attribute",
                    "DS[calc viral attribute At := expr]",
                    (authored(),),
                ),
                Variant(
                    "multiple",
                    "DS[calc Me := expr, attribute At := expr]",
                    (authored(), authored("minimal_2")),
                ),
            ),
        ),
        Operator(
            id="aggr",
            name="Aggregation",
            symbol="aggr",
            page="Clause operators/Aggregation",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                Variant("group_by", "DS[aggr Me := agg(Me) group by Id]", (manual("ex_1"),)),
                Variant(
                    "group_except",
                    "DS[aggr Me := agg(Me) group except Id]",
                    (manual("ex_2"),),
                ),
                Variant(
                    "group_all",
                    'DS[aggr Me := agg(Me) group all time_agg("A")]',
                    (authored(),),
                ),
                Variant(
                    "having",
                    "DS[aggr Me := agg(Me) group by Id having cond]",
                    (manual("ex_3"),),
                ),
                Variant(
                    "attribute_role",
                    "DS[aggr attribute At := agg(Me) group by Id]",
                    (authored(),),
                ),
                Variant(
                    "viral_attribute_role",
                    "DS[aggr viral attribute At := agg(Me) group by Id]",
                    (authored(),),
                ),
            ),
        ),
        Operator(
            id="filter",
            name="Filtering Data Points",
            symbol="filter",
            page="Clause operators/Filtering Data Points",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(Variant("basic", "DS[filter cond]", (manual("ex_1"), authored())),),
        ),
        Operator(
            id="keep",
            name="Maintaining Components",
            symbol="keep",
            page="Clause operators/Maintaining Components",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(Variant("basic", "DS[keep comp]", (manual("ex_1"),)),),
        ),
        Operator(
            id="drop",
            name="Removal of Components",
            symbol="drop",
            page="Clause operators/Removal of Components",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(Variant("basic", "DS[drop comp]", (manual("ex_1"),)),),
        ),
        Operator(
            id="rename",
            name="Change of Component name",
            symbol="rename",
            page="Clause operators/Change of Component name",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(
                Variant(
                    "basic",
                    "DS[rename comp to comp]",
                    (authored(), authored("minimal_2")),
                ),
                Variant(
                    "multiple",
                    "DS[rename comp to comp, comp to comp]",
                    (manual("ex_1"),),
                ),
            ),
        ),
        Operator(
            id="pivot",
            name="Pivoting",
            symbol="pivot",
            page="Clause operators/Pivoting",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(Variant("basic", "DS[pivot Id, Me]", (manual("ex_1"), authored())),),
        ),
        Operator(
            id="unpivot",
            name="Unpivoting",
            symbol="unpivot",
            page="Clause operators/Unpivoting",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(
                Variant(
                    "basic",
                    "DS[unpivot Id, Me]",
                    (manual("ex_1"), manual("ex_2"), authored()),
                ),
            ),
        ),
        Operator(
            id="sub",
            name="Subspace",
            symbol="sub",
            page="Clause operators/Subspace",
            accepts=("dataset", "scalar"),
            returns=("dataset",),
            variants=(
                Variant("one_id", "DS[sub Id = value]", (manual("ex_3"), authored())),
                Variant(
                    "multi_id",
                    "DS[sub Id = value, Id = value]",
                    (manual("ex_1"), manual("ex_2")),
                ),
            ),
        ),
    ),
)
