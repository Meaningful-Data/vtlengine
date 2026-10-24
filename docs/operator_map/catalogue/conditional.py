"""The conditional operators of the VTL 2.2 Reference Manual (chapter ``Conditional operators``).

A variant is one syntax form: the level of each operand (``DS`` a dataset, ``comp`` a component
inside ``calc``, ``sc`` a scalar), whether a branch of ``if-then-else`` and ``case`` is a scalar
(``scalar_branch``), and for ``case`` how many ``when`` clauses it has (``multi_when``). Each form
says what it takes and gives, by the "Input parameters type" and "Result type" of its page.

``if-then-else`` and ``case`` take a boolean condition (``dataset { measure <boolean> _ } |
component<Boolean> | boolean``) and branches that are a dataset, a component or a scalar of any
type, the same for both ("must be of the same scalar type"). ``nvl`` takes two operands, each a
dataset, a component or a scalar of any type, again the same for both. The result of every form
is a dataset, a component or a scalar of the type of its operands: it follows its input, so every
form gives ``any``.
"""

from __future__ import annotations

from tool.model import ANY_SCALAR, ANY_TYPE, Chapter, Operator, Variant, authored, manual

_PAGE = "Conditional operators"

# What a form takes and gives. The level of its operands, a dataset, a component or a scalar, is
# its kind and not a type, so a form on datasets has the types of the same form on scalars.
_CONDITION_AND_BRANCHES = ("boolean", ANY_SCALAR)  # the condition, then thenOperand and elseOperand
_OPERANDS = (ANY_SCALAR,)  # op1 and op2 of nvl
_GIVES = (ANY_TYPE,)  # the result follows the operands

# The types of an operator: those of its forms, and ``dataset`` for its forms on datasets.
_OP_CONDITIONAL = (*_CONDITION_AND_BRANCHES, "dataset")
_OP_NVL = (*_OPERANDS, "dataset")
_OP_GIVES = (*_GIVES, "dataset")

CHAPTER = Chapter(
    id="conditional",
    title=_PAGE,
    operators=(
        Operator(
            id="if_then_else",
            name="if-then-else",
            symbol="if then else",
            page=f"{_PAGE}/if-then-else",
            accepts=_OP_CONDITIONAL,
            returns=_OP_GIVES,
            variants=(
                Variant(
                    "ds",
                    "if DS then DS else DS",
                    (manual("ex_1"),),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
                Variant(
                    "ds_scalar_branch",
                    "if DS then DS|sc else sc|DS",
                    (
                        authored("scalar_else_feeds_all_measures"),
                        authored("scalar_then_feeds_all_measures"),
                    ),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
                Variant(
                    "comp",
                    "if comp then comp else comp",
                    (authored("larger_of_two_components"),),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
                Variant(
                    "comp_scalar_branch",
                    "if comp then sc else sc",
                    (authored("flag_from_a_threshold"),),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
                Variant(
                    "sc",
                    "if sc then sc else sc",
                    (
                        authored("true_condition_gives_then_operand"),
                        authored("false_condition_gives_else_operand"),
                    ),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
            ),
        ),
        Operator(
            id="case",
            name="Case",
            symbol="case when",
            page=f"{_PAGE}/Case",
            accepts=_OP_CONDITIONAL,
            returns=_OP_GIVES,
            variants=(
                Variant(
                    "ds",
                    "case when DS then DS else DS",
                    (
                        authored("inline_dataset_condition"),
                        authored("unmatched_data_point_left_out"),
                    ),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
                Variant(
                    "ds_scalar_branch",
                    "case when DS then DS|sc else sc|DS",
                    (
                        authored("scalar_else_feeds_all_measures"),
                        authored("scalar_then_feeds_all_measures"),
                    ),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
                Variant(
                    "ds_multi_when",
                    "case when DS then DS when DS then DS ... else DS",
                    (
                        authored("overlapping_inline_conditions"),
                        authored("overlapping_named_conditions"),
                    ),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
                Variant(
                    "comp",
                    "case when comp then comp else comp",
                    (authored("smaller_of_two_components"),),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
                Variant(
                    "comp_multi_when",
                    "case when comp then sc when comp then sc ... else sc",
                    (manual("ex_1"), authored("overlapping_descending_thresholds")),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
                Variant(
                    "sc",
                    "case when sc then sc else sc",
                    (
                        authored("true_condition_gives_then_operand"),
                        authored("false_condition_gives_else_operand"),
                    ),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
                Variant(
                    "sc_multi_when",
                    "case when sc then sc when sc then sc ... else sc",
                    (
                        authored("first_condition_true"),
                        authored("second_condition_true"),
                        authored("no_condition_true"),
                        authored("both_conditions_true"),
                    ),
                    takes=_CONDITION_AND_BRANCHES,
                    gives=_GIVES,
                ),
            ),
        ),
        Operator(
            id="nvl",
            name="Nvl",
            symbol="nvl",
            page=f"{_PAGE}/Nvl",
            accepts=_OP_NVL,
            returns=_OP_GIVES,
            variants=(
                Variant("ds_sc", "nvl(DS, sc)", (manual("ex_1"),), takes=_OPERANDS, gives=_GIVES),
                Variant(
                    "ds_ds",
                    "nvl(DS, DS)",
                    (authored("partly_different_identifiers"),),
                    takes=_OPERANDS,
                    gives=_GIVES,
                ),
                Variant(
                    "sc_ds",
                    "nvl(sc, DS)",
                    (authored("non_null_scalar_first_operand"),),
                    takes=_OPERANDS,
                    gives=_GIVES,
                ),
                Variant(
                    "comp_sc",
                    "nvl(comp, sc)",
                    (authored("null_replaced_by_zero"),),
                    takes=_OPERANDS,
                    gives=_GIVES,
                ),
                Variant(
                    "comp_comp",
                    "nvl(comp, comp)",
                    (authored("second_component_fills_nulls"),),
                    takes=_OPERANDS,
                    gives=_GIVES,
                ),
                Variant(
                    "sc_comp",
                    "nvl(sc, comp)",
                    (authored("non_null_scalar_first_operand"),),
                    takes=_OPERANDS,
                    gives=_GIVES,
                ),
                Variant(
                    "sc_sc",
                    "nvl(sc, sc)",
                    (authored("null_first_operand"), authored("non_null_first_operand")),
                    takes=_OPERANDS,
                    gives=_GIVES,
                ),
            ),
        ),
    ),
)
