"""Hierarchical aggregation: the ``hierarchy`` operator (hierarchical roll-up).

The variants are the grid ``<mode>_<input>_<output>`` of the manual's syntax, every value of the
three parameters written out, plus the two clauses that bind the ruleset to the operand:
``condition`` and the ``rule`` clause of a ruleset defined on value domains. The ``minimal``
examples of the grid share one input dataset and one ruleset (``A = B + C; C = D + E``, defined
on the variable ``Id_2``), so each expected output follows from the mode table of the page. The
``rule_priority`` variants also have ``minimal_2``: the same input with a C in ``DS_1`` where the
rule computes a C that is not null. The manual examples differ only in their mode, with the
default input and output, so each one goes to the grid variant of its mode.
"""

from __future__ import annotations

from tool.model import Chapter, Example, Operator, Variant, authored, manual

_PAGE = "Hierarchical aggregation/Hierarchical roll-up"


# ``rule_priority`` also runs on the grid input plus a C of ``DS_1`` where the rule computes a C
# that is not null, so that the computed C coming first is told from the C of ``DS_1`` coming first.
_PRECEDENCE: tuple[Example, ...] = (authored("minimal_2"),)


def _grid(
    mode: str, input_: str, output: str, *examples: Example, extra: tuple[Example, ...] = ()
) -> Variant:
    """The grid variant ``<mode>_<input>_<output>``: any manual examples, ``minimal``, ``extra``."""
    return Variant(
        id=f"{mode}_{input_}_{output}",
        label=f"hierarchy(DS, HR {mode} {input_} {output})",
        examples=(*examples, authored(), *extra),
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
            examples=(authored(),),
        ),
        Variant(
            id="valuedomain_rule",
            label="hierarchy(DS, HR rule R)",
            examples=(authored(),),
        ),
    ),
)

CHAPTER = Chapter(id="hierarchy", title="Hierarchical aggregation", operators=(HIERARCHY,))
