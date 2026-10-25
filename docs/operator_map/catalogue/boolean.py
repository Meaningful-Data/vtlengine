"""The Boolean operators of the VTL 2.2 Reference Manual (chapter ``Boolean operators``).

A variant is one syntax form: the level of each operand (``DS`` a dataset, ``comp`` a component
inside ``calc``, ``sc`` a scalar). The binary operators have the seven pairs of operand levels,
and ``not`` takes one operand. Each form says what it takes and gives, by the "Input parameters
type" and "Result type" of its page: ``dataset {measure<boolean> _ } | component<boolean> |
boolean``, for the operands and for the result of the four operators.

The manual examples are the two-valued ones: ``ex_1`` is the dataset pair and ``ex_2`` the
component and scalar pair. The authored examples bring in the three-valued logic, the null values.
"""

from __future__ import annotations

from tool.model import Chapter, Example, Operator, Variant, authored, manual

# What a form takes and gives. The level of its operands, a dataset, a component or a scalar, is
# its kind and not a type, so ``DS and DS`` has the types of ``sc and sc``.
_BOOLEAN = ("boolean",)

# The types of an operator: those of its forms, and ``dataset`` for its forms on datasets.
_OP_BOOLEAN = (*_BOOLEAN, "dataset")


def _form(variant_id: str, label: str, *examples: Example) -> Variant:
    """A form of a Boolean operator: it takes booleans and gives a boolean."""
    return Variant(variant_id, label, examples, takes=_BOOLEAN, gives=_BOOLEAN)


CHAPTER = Chapter(
    id="boolean",
    title="Boolean operators",
    operators=(
        Operator(
            id="and",
            name="Logical conjunction",
            symbol="and",
            page="Boolean operators/Logical conjunction",
            accepts=_OP_BOOLEAN,
            returns=_OP_BOOLEAN,
            variants=(
                _form(
                    "ds_ds",
                    "DS and DS",
                    manual("ex_1"),
                    authored("null_operands_and_uncoupled_point"),
                ),
                _form(
                    "ds_sc",
                    "DS and sc",
                    authored("true_keeps_the_other_operand"),
                    authored("false_decides_even_over_null"),
                ),
                _form(
                    "sc_ds",
                    "sc and DS",
                    authored("true_keeps_the_other_operand"),
                    authored("false_decides_even_over_null"),
                ),
                _form(
                    "comp_comp",
                    "comp and comp",
                    authored("all_four_boolean_combinations"),
                    authored("null_on_either_side"),
                ),
                _form(
                    "comp_sc",
                    "comp and sc",
                    manual("ex_2"),
                    authored("true_keeps_the_other_operand"),
                    authored("false_decides_even_over_null"),
                ),
                _form(
                    "sc_comp",
                    "sc and comp",
                    authored("true_keeps_the_other_operand"),
                    authored("false_decides_even_over_null"),
                ),
                _form(
                    "sc_sc",
                    "sc and sc",
                    authored("all_four_boolean_combinations"),
                    authored("one_null_operand_on_either_side"),
                    authored("both_operands_null"),
                ),
            ),
        ),
        Operator(
            id="or",
            name="Logical disjunction",
            symbol="or",
            page="Boolean operators/Logical disjunction",
            accepts=_OP_BOOLEAN,
            returns=_OP_BOOLEAN,
            variants=(
                _form(
                    "ds_ds",
                    "DS or DS",
                    manual("ex_1"),
                    authored("null_operands_and_uncoupled_point"),
                ),
                _form(
                    "ds_sc",
                    "DS or sc",
                    authored("false_keeps_the_other_operand"),
                    authored("true_decides_even_over_null"),
                ),
                _form(
                    "sc_ds",
                    "sc or DS",
                    authored("false_keeps_the_other_operand"),
                    authored("true_decides_even_over_null"),
                ),
                _form(
                    "comp_comp",
                    "comp or comp",
                    authored("all_four_boolean_combinations"),
                    authored("null_on_either_side"),
                ),
                _form(
                    "comp_sc",
                    "comp or sc",
                    manual("ex_2"),
                    authored("false_keeps_the_other_operand"),
                    authored("true_decides_even_over_null"),
                ),
                _form(
                    "sc_comp",
                    "sc or comp",
                    authored("false_keeps_the_other_operand"),
                    authored("true_decides_even_over_null"),
                ),
                _form(
                    "sc_sc",
                    "sc or sc",
                    authored("all_four_boolean_combinations"),
                    authored("one_null_operand_on_either_side"),
                    authored("both_operands_null"),
                ),
            ),
        ),
        Operator(
            id="xor",
            name="Exclusive disjunction",
            symbol="xor",
            page="Boolean operators/Exclusive disjunction",
            accepts=_OP_BOOLEAN,
            returns=_OP_BOOLEAN,
            variants=(
                _form(
                    "ds_ds",
                    "DS xor DS",
                    manual("ex_1"),
                    authored("null_operands_and_uncoupled_point"),
                ),
                _form("ds_sc", "DS xor sc", authored("true_flips_and_keeps_null")),
                _form("sc_ds", "sc xor DS", authored("true_flips_and_keeps_null")),
                _form(
                    "comp_comp",
                    "comp xor comp",
                    authored("all_four_boolean_combinations"),
                    authored("null_on_either_side"),
                ),
                _form(
                    "comp_sc",
                    "comp xor sc",
                    manual("ex_2"),
                    authored("true_flips_and_keeps_null"),
                ),
                _form("sc_comp", "sc xor comp", authored("true_flips_and_keeps_null")),
                _form(
                    "sc_sc",
                    "sc xor sc",
                    authored("all_four_boolean_combinations"),
                    authored("one_null_operand_on_either_side"),
                    authored("both_operands_null"),
                ),
            ),
        ),
        Operator(
            id="not",
            name="Logical negation",
            symbol="not",
            page="Boolean operators/Logical negation",
            accepts=_OP_BOOLEAN,
            returns=_OP_BOOLEAN,
            variants=(
                _form("ds", "not DS", manual("ex_1"), authored("true_false_and_null_values")),
                _form("comp", "not comp", manual("ex_2"), authored("true_false_and_null_values")),
                _form(
                    "sc",
                    "not sc",
                    authored("true_and_false_operands"),
                    authored("null_operand"),
                ),
            ),
        ),
    ),
)
