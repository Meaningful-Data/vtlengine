"""Hierarchical aggregation: the ``hierarchy`` operator (hierarchical roll-up).

The variants are the grid ``<mode>_<input>_<output>`` of the manual's syntax, every value of the
three parameters written out, plus the two clauses that bind the ruleset to the operand:
``condition`` and the ``rule`` clause of a ruleset defined on value domains. The authored grid
examples share one input dataset and one ruleset (``A = B + C; C = D + E``, defined on the
variable ``Id_2``), so each expected output follows from the mode table of the page. The manual
examples differ only in their mode, with the default input and output, so each one goes to the
grid variant of its mode.
"""

from __future__ import annotations

from tool.model import Chapter, Example, Operator, Variant, authored, manual

_PAGE = "Hierarchical aggregation/Hierarchical roll-up"


def _grid(mode: str, input_: str, output: str, *examples: Example) -> Variant:
    """The grid variant ``<mode>_<input>_<output>``: its authored example, after any others."""
    return Variant(
        id=f"{mode}_{input_}_{output}",
        label=f"hierarchy(DS, HR {mode} {input_} {output})",
        examples=(*examples, authored()),
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
        _grid("non_null", "rule_priority", "computed"),
        _grid("non_null", "rule_priority", "all"),
        _grid("non_zero", "dataset", "computed"),
        _grid("non_zero", "dataset", "all"),
        _grid("non_zero", "rule", "computed", manual("ex_2")),
        _grid("non_zero", "rule", "all"),
        _grid("non_zero", "rule_priority", "computed"),
        _grid("non_zero", "rule_priority", "all"),
        _grid("partial_null", "dataset", "computed"),
        _grid("partial_null", "dataset", "all"),
        _grid("partial_null", "rule", "computed", manual("ex_3")),
        _grid("partial_null", "rule", "all"),
        _grid("partial_null", "rule_priority", "computed"),
        _grid("partial_null", "rule_priority", "all"),
        _grid("partial_zero", "dataset", "computed"),
        _grid("partial_zero", "dataset", "all"),
        _grid("partial_zero", "rule", "computed"),
        _grid("partial_zero", "rule", "all"),
        _grid("partial_zero", "rule_priority", "computed"),
        _grid("partial_zero", "rule_priority", "all"),
        _grid("always_null", "dataset", "computed"),
        _grid("always_null", "dataset", "all"),
        _grid("always_null", "rule", "computed"),
        _grid("always_null", "rule", "all"),
        _grid("always_null", "rule_priority", "computed"),
        _grid("always_null", "rule_priority", "all"),
        _grid("always_zero", "dataset", "computed"),
        _grid("always_zero", "dataset", "all"),
        _grid("always_zero", "rule", "computed"),
        _grid("always_zero", "rule", "all"),
        _grid("always_zero", "rule_priority", "computed"),
        _grid("always_zero", "rule_priority", "all"),
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
