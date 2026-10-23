"""Set operators: the union, intersection and differences of data sets.

The four operators take data sets only: they are not defined on scalars or components. Their
operands have the same Identifier, Measure and Attribute Components, and the data points of the
operands are compared by their Identifier values. ``union`` and ``intersect`` take two or more
operands, ``setdiff`` and ``symdiff`` exactly two.
"""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

CHAPTER = Chapter(
    id="set",
    title="Set operators",
    operators=(
        Operator(
            id="union",
            name="Union",
            symbol="union",
            page="Set operators/Union",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(
                Variant(
                    id="two",
                    label="union(DS, DS)",
                    examples=(manual("ex_1"), manual("ex_2")),
                ),
                Variant(
                    id="n",
                    label="union(DS, DS, ...)",
                    examples=(authored(),),
                ),
            ),
        ),
        Operator(
            id="intersect",
            name="Intersection",
            symbol="intersect",
            page="Set operators/Intersection",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(
                Variant(
                    id="two",
                    label="intersect(DS, DS)",
                    examples=(manual("ex_1"),),
                ),
                Variant(
                    id="n",
                    label="intersect(DS, DS, ...)",
                    examples=(authored(),),
                ),
            ),
        ),
        Operator(
            id="setdiff",
            name="Set difference",
            symbol="setdiff",
            page="Set operators/Set difference",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(
                Variant(
                    id="two",
                    label="setdiff(DS, DS)",
                    examples=(manual("ex_1"), manual("ex_2")),
                ),
            ),
        ),
        Operator(
            id="symdiff",
            name="Symmetric difference",
            symbol="symdiff",
            page="Set operators/Symmetric difference",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(
                Variant(
                    id="two",
                    label="symdiff(DS, DS)",
                    examples=(manual("ex_1"),),
                ),
            ),
        ),
    ),
)
