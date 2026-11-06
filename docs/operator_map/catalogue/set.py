"""Set operators: the union, intersection and differences of data sets.

The four operators take data sets only: they are not defined on scalars or components. Their
operands have the same Identifier, Measure and Attribute Components, and the data points of the
operands are compared by their Identifier values. ``union`` and ``intersect`` take two or more
operands, ``setdiff`` and ``symdiff`` exactly two.

Each form says what it takes and gives, by the "Input parameters type" and "Result type" of its
page. The four pages type the operands and the result ``dataset``, with no type for the measures,
so a form takes data sets of any type, as ``scalar`` (``any`` is only for what a form gives), and
gives a data set whose types follow its operands (``any``). No form has a kind: no id carries a
kind token, and every form works on data sets.
"""

from __future__ import annotations

from tool.model import ANY_SCALAR, ANY_TYPE, Chapter, Example, Operator, Variant, authored, manual

# What a form takes and gives: its operands, data sets of any type, as ``scalar``; and a data set
# whose types follow its operands, as ``any``.
_DATA_SETS = (ANY_SCALAR,)
_GIVES = (ANY_TYPE,)


def _form(variant_id: str, label: str, *examples: Example) -> Variant:
    """A form of a set operator: it takes data sets and gives a data set that follows them."""
    return Variant(variant_id, label, examples, takes=_DATA_SETS, gives=_GIVES)


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
                _form(
                    "two",
                    "union(DS, DS)",
                    manual("ex_1"),
                    manual("ex_2"),
                    authored("leftmost_wins_when_measures_differ"),
                ),
                _form(
                    "n",
                    "union(DS, DS, ...)",
                    authored("leftmost_of_three_operands_wins"),
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
                _form(
                    "two",
                    "intersect(DS, DS)",
                    manual("ex_1"),
                    authored("leftmost_wins_when_measures_differ"),
                ),
                _form(
                    "n",
                    "intersect(DS, DS, ...)",
                    authored("point_in_all_three_operands"),
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
                _form(
                    "two",
                    "setdiff(DS, DS)",
                    manual("ex_1"),
                    manual("ex_2"),
                    authored("matched_by_identifiers_not_measures"),
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
                _form(
                    "two",
                    "symdiff(DS, DS)",
                    manual("ex_1"),
                    authored("matched_by_identifiers_not_measures"),
                ),
            ),
        ),
    ),
)
