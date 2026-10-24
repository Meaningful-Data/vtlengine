:Tests: a parameter with a condition, ``x integer [value >= 0]``, called with ``2``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``scalarTypeConstraint ::= [ valueBooleanCondition ] | ...``, a condition "expressed with reference to the fictitious variable" ``value``, as in ``number [ value >= 0 ]``. The script ends with ``end operator``, as the page's examples and the grammar do; its syntax line writes ``end define operator``.
:Expected: ``SC_r = 20``, an Integer: ``2`` meets the condition, and ``2 * 10 = 20``.
:Finding: both engines raise SemanticError ``1-3-13``, "Unsupported VTL construct: type constraint in 'integer [value >= 0]'".
