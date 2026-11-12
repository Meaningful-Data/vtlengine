"""Clause operators: the clauses written in square brackets after a data set, ``DS[...]``.

Each clause takes one data set and returns a data set; none of them applies to scalars. ``calc``,
``aggr`` and ``filter`` hold expressions at component level; ``keep``, ``drop``, ``rename``,
``pivot`` and ``unpivot`` name components; ``sub`` fixes identifiers to scalar values.

Every form takes its operand, a data set of any type, as ``scalar``, and gives a data set whose
types follow it (``any``). A form also takes the expressions of its clause, as the pages type
them: a ``filter`` or ``having`` condition is a boolean; a ``calc`` or ``aggr`` expression, the
``conversionExpr`` of ``group all`` and the ``value`` of ``sub`` are any scalar. The names of the
components that a clause calculates, keeps, drops, renames, groups by or fixes, and the identifier
and the measure of ``pivot`` and ``unpivot``, add no type. No form has a kind: no id carries a
kind token, and every form works on a data set.

``calc identifier`` over an expression declared nullable raises 1-1-1-16 by the team decision
of #1010, so the ``identifier`` example declares its expression non-nullable. The ``basic`` form
of ``rename`` has an example that expects the error of the first error condition its page
states: a new name that is already the name of a component of the operand.
"""

from __future__ import annotations

from typing import Final

from tool.model import ANY_SCALAR, ANY_TYPE, Chapter, Example, Operator, Variant, authored, manual

# What a form takes ("Input parameters type" of each page): its operand, a data set of any type,
# as ``scalar`` (``any`` is only for what a form gives: its result follows its operand), and the
# expressions of its clause.
_OPERAND: Final = (ANY_SCALAR,)  # op: dataset
_CALC: Final = (ANY_SCALAR,)  # calcExpr: component<scalar>
_AGGR: Final = (ANY_SCALAR,)  # aggrExpr: component<scalar>; conversionExpr: identifier<scalar>
_HAVING: Final = (ANY_SCALAR, "boolean")  # and the havingCondition: component<boolean>
_FILTER: Final = (ANY_SCALAR, "boolean")  # filterCondition: component<boolean>
_SUB: Final = (ANY_SCALAR,)  # value: scalar

# The page takes the Measures of the result from the data ("each distinct value of identifier in
# op") and types Id_2 String, but the structure must be known without the data: the Measures are
# the values of the value domain of the Identifier (#1234; Javier, 2026-11-12).
_PIVOT_ERRATUM: Final = (
    "the Measures come from the data (each distinct value of identifier) and ds_1.json types Id_2 "
    "String, but they must be known without the data: Id_2 is defined on the value domain VD_1 "
    "of A, B and C, whose values give the Measures"
)


def _form(variant_id: str, label: str, takes: tuple[str, ...], *examples: Example) -> Variant:
    """A form of a clause: it takes ``takes`` and gives a data set whose types follow its input."""
    return Variant(id=variant_id, label=label, examples=examples, takes=takes, gives=(ANY_TYPE,))


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
                _form(
                    "measure_default",
                    "DS[calc Me := expr]",
                    _CALC,
                    manual("ex_1"),
                    authored("new_component_is_measure"),
                    authored("existing_attribute_keeps_role"),
                ),
                _form(
                    "measure_explicit",
                    "DS[calc measure Me := expr]",
                    _CALC,
                    authored("explicit_role_on_new_component"),
                    authored("attribute_changed_to_measure"),
                ),
                _form(
                    "identifier",
                    "DS[calc identifier Id := expr]",
                    _CALC,
                    authored("new_identifier_from_measure"),
                ),
                _form("attribute", "DS[calc attribute At := expr]", _CALC, manual("ex_2")),
                _form(
                    "viral_attribute",
                    "DS[calc viral attribute At := expr]",
                    _CALC,
                    authored("constant_viral_attribute"),
                ),
                _form(
                    "multiple",
                    "DS[calc Me := expr, attribute At := expr]",
                    _CALC,
                    authored("default_and_explicit_roles"),
                    authored("items_are_independent"),
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
                _form(
                    "no_group",
                    "DS[aggr Me := agg(Me)]",
                    _AGGR,
                    authored("single_group_no_identifiers"),
                ),
                _form("group_by", "DS[aggr Me := agg(Me) group by Id]", _AGGR, manual("ex_1")),
                _form(
                    "group_except",
                    "DS[aggr Me := agg(Me) group except Id]",
                    _AGGR,
                    manual("ex_2"),
                ),
                _form(
                    "group_all",
                    'DS[aggr Me := agg(Me) group all time_agg("A")]',
                    _AGGR,
                    authored("quarters_to_years"),
                ),
                _form(
                    "having",
                    "DS[aggr Me := agg(Me) group by Id having cond]",
                    _HAVING,
                    manual("ex_3"),
                ),
                _form(
                    "measure_role",
                    "DS[aggr measure Me := agg(Me) group by Id]",
                    _AGGR,
                    authored("only_the_new_measure_kept"),
                ),
                _form(
                    "attribute_role",
                    "DS[aggr attribute At := agg(Me) group by Id]",
                    _AGGR,
                    authored("attribute_next_to_measure"),
                    authored("attribute_without_measure"),
                ),
                _form(
                    "viral_attribute_role",
                    "DS[aggr viral attribute At := agg(Me) group by Id]",
                    _AGGR,
                    authored("viral_attribute_next_to_measure"),
                    authored("viral_attribute_without_measure"),
                ),
                _form(
                    "role_omitted_attribute",
                    "DS[aggr At := agg(At) group by Id]",
                    _AGGR,
                    authored("existing_attribute_keeps_role"),
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
            variants=(
                _form(
                    "basic",
                    "DS[filter cond]",
                    _FILTER,
                    manual("ex_1"),
                    authored("null_condition_filtered_out"),
                ),
            ),
        ),
        Operator(
            id="keep",
            name="Maintaining Components",
            symbol="keep",
            page="Clause operators/Maintaining Components",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(_form("basic", "DS[keep comp]", _OPERAND, manual("ex_1")),),
        ),
        Operator(
            id="drop",
            name="Removal of Components",
            symbol="drop",
            page="Clause operators/Removal of Components",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(_form("basic", "DS[drop comp]", _OPERAND, manual("ex_1")),),
        ),
        Operator(
            id="rename",
            name="Change of Component name",
            symbol="rename",
            page="Clause operators/Change of Component name",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(
                _form(
                    "basic",
                    "DS[rename comp to comp]",
                    _OPERAND,
                    authored("measure_renamed"),
                    authored("identifier_renamed"),
                    authored("new_name_already_in_operand"),
                ),
                _form(
                    "multiple",
                    "DS[rename comp to comp, comp to comp]",
                    _OPERAND,
                    manual("ex_1"),
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
            variants=(
                _form(
                    "basic",
                    "DS[pivot Id, Me]",
                    _OPERAND,
                    manual("ex_1", erratum=_PIVOT_ERRATUM),
                    authored("missing_combination_is_null"),
                ),
            ),
        ),
        Operator(
            id="unpivot",
            name="Unpivoting",
            symbol="unpivot",
            page="Clause operators/Unpivoting",
            accepts=("dataset",),
            returns=("dataset",),
            variants=(
                _form(
                    "basic",
                    "DS[unpivot Id, Me]",
                    _OPERAND,
                    manual("ex_1"),
                    manual("ex_2"),
                    authored("null_measure_skipped"),
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
                _form(
                    "one_id",
                    "DS[sub Id = value]",
                    _SUB,
                    manual("ex_3"),
                    authored("fixed_identifier_dropped"),
                ),
                _form(
                    "multi_id",
                    "DS[sub Id = value, Id = value]",
                    _SUB,
                    manual("ex_1"),
                    manual("ex_2"),
                ),
            ),
        ),
    ),
)
