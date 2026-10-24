"""Hierarchical aggregation: the ``hierarchy`` operator (hierarchical roll-up).

The variants are the grid ``<mode>_<input>_<output>`` of the manual's syntax, every value of the
three parameters written out, plus the two clauses that bind the ruleset to the operand:
``condition`` and the ``rule`` clause of a ruleset defined on value domains.

Every form takes its operand, a data set whose measures are numbers
(``op :: dataset { measure<number> _ }``), as ``number``, and the ruleset
(``hr :: name<hierarchical>``) as ``hierarchical_ruleset``, the type that
``define hierarchical ruleset`` gives. The components that ``condition`` and ``rule`` name are
names, with no type. Every form gives a data set with the identifiers and the measure of its
operand: its result follows its input (``any``). No form has a kind: no id carries a kind token,
and every form works on data sets.

The ``missing_null_and_zero_items`` examples of the grid share one input dataset and one ruleset
(``A = B + C; C = D + E``, defined on the variable ``Id_2``), so each expected output follows from
the mode, input and output texts of the page and its mode table. Each group of ``Id_1`` has its own
job: 1 has no C, which tells ``dataset`` from ``rule`` and ``rule_priority``; 2 has a null D beside
a C of 20, which tells ``rule`` from ``rule_priority``; 3 has only B = 0, which tells the modes
apart. So every mode gives a different result for each input and output, and the three inputs
give three different results in every mode. The ``rule_priority`` variants also have
``computed_item_takes_precedence``: the same input with a C in ``DS_1`` where the ruleset computes
a C that is not null. The manual examples differ only in their mode, with the default input and
output, so each one goes to the grid variant of its mode.

The ``condition`` variant also has ``overlapping_rules_that_agree``: two rules whose conditions
overlap compute the same code item for one data point, and agree on its value. The manual does not
say what the result is; by the team decision of #1193 the data point is returned once.
"""

from __future__ import annotations

from typing import Final

from tool.model import ANY_TYPE, Chapter, Example, Operator, Variant, authored, manual

_PAGE = "Hierarchical aggregation/Hierarchical roll-up"

# What every form takes and gives ("Input parameters type" and "Result type" of the page): the
# numbers of the operand's measures and the ruleset; a data set whose types follow the operand.
_TAKES: Final = ("number", "hierarchical_ruleset")
_GIVES: Final = (ANY_TYPE,)

# The shared input of the grid: a missing C, a null D and a lone zero B, one per group of Id_1.
_GRID_EXAMPLE: Final = "missing_null_and_zero_items"
# ``rule_priority`` also runs on the grid input plus a C of ``DS_1`` where the rule computes a C
# that is not null, so that the computed C coming first is told from the C of ``DS_1`` coming first.
_PRECEDENCE: tuple[Example, ...] = (authored("computed_item_takes_precedence"),)
# Two rules that compute the same code item for a data point, with the same value: once (#1193).
_OVERLAP_DECISION = 1193


def _grid(
    mode: str, input_: str, output: str, *examples: Example, extra: tuple[Example, ...] = ()
) -> Variant:
    """The grid variant ``<mode>_<input>_<output>``: any manual examples, the grid's, ``extra``."""
    return Variant(
        id=f"{mode}_{input_}_{output}",
        label=f"hierarchy(DS, HR {mode} {input_} {output})",
        examples=(*examples, authored(_GRID_EXAMPLE), *extra),
        takes=_TAKES,
        gives=_GIVES,
    )


HIERARCHY = Operator(
    id="hierarchy",
    name="Hierarchical roll-up",
    symbol="hierarchy",
    page=_PAGE,
    accepts=("dataset", "number", "hierarchical_ruleset"),
    returns=("dataset",),
    variants=(
        _grid("non_null", "dataset", "computed"),
        _grid("non_null", "dataset", "all"),
        _grid("non_null", "rule", "computed", manual("ex_1")),
        _grid("non_null", "rule", "all"),
        _grid("non_null", "rule_priority", "computed", extra=_PRECEDENCE),
        _grid("non_null", "rule_priority", "all", extra=_PRECEDENCE),
        _grid("non_zero", "dataset", "computed"),
        _grid("non_zero", "dataset", "all"),
        _grid("non_zero", "rule", "computed", manual("ex_2")),
        _grid("non_zero", "rule", "all"),
        _grid("non_zero", "rule_priority", "computed", extra=_PRECEDENCE),
        _grid("non_zero", "rule_priority", "all", extra=_PRECEDENCE),
        _grid("partial_null", "dataset", "computed"),
        _grid("partial_null", "dataset", "all"),
        _grid("partial_null", "rule", "computed", manual("ex_3")),
        _grid("partial_null", "rule", "all"),
        _grid("partial_null", "rule_priority", "computed", extra=_PRECEDENCE),
        _grid("partial_null", "rule_priority", "all", extra=_PRECEDENCE),
        _grid("partial_zero", "dataset", "computed"),
        _grid("partial_zero", "dataset", "all"),
        _grid("partial_zero", "rule", "computed"),
        _grid("partial_zero", "rule", "all"),
        _grid("partial_zero", "rule_priority", "computed", extra=_PRECEDENCE),
        _grid("partial_zero", "rule_priority", "all", extra=_PRECEDENCE),
        _grid("always_null", "dataset", "computed"),
        _grid("always_null", "dataset", "all"),
        _grid("always_null", "rule", "computed"),
        _grid("always_null", "rule", "all"),
        _grid("always_null", "rule_priority", "computed", extra=_PRECEDENCE),
        _grid("always_null", "rule_priority", "all", extra=_PRECEDENCE),
        _grid("always_zero", "dataset", "computed"),
        _grid("always_zero", "dataset", "all"),
        _grid("always_zero", "rule", "computed"),
        _grid("always_zero", "rule", "all"),
        _grid("always_zero", "rule_priority", "computed", extra=_PRECEDENCE),
        _grid("always_zero", "rule_priority", "all", extra=_PRECEDENCE),
        Variant(
            id="condition",
            label="hierarchy(DS, HR condition C)",
            examples=(
                authored("rule_only_where_the_condition_holds"),
                authored("overlapping_rules_that_agree", decision=_OVERLAP_DECISION),
            ),
            takes=_TAKES,
            gives=_GIVES,
        ),
        Variant(
            id="valuedomain_rule",
            label="hierarchy(DS, HR rule R)",
            examples=(authored("domain_ruleset_bound_by_rule"),),
            takes=_TAKES,
            gives=_GIVES,
        ),
    ),
)

CHAPTER = Chapter(id="hierarchy", title="Hierarchical aggregation", operators=(HIERARCHY,))
