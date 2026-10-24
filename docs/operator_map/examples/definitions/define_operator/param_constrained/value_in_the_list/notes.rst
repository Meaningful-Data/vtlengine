:Tests: a parameter restricted to a list of values, ``x integer { 1, 2, 3 }``, called with ``2``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``scalarTypeConstraint ::= ... | { scalarLiteral { , scalarLiteral }* }``, as in "integer { 0, 1 } means an integer number whose value is 0 or 1". The script ends with ``end operator``, as the page's examples do; its syntax line writes ``end define operator``.
:Expected: ``SC_r = 20``, an Integer: ``2`` is in the list, and ``2 * 10 = 20``.
:Finding: both engines raise SemanticError ``1-3-13``, "Unsupported VTL construct: type constraint in 'integer { 1, 2, 3 }'".
