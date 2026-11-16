:Tests: a parameter restricted to a list of values, ``x integer { 1, 2, 3 }``, called as ``times_ten(SC_1)`` with ``SC_1`` = 2.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``scalarTypeConstraint ::= ... | { scalarLiteral { , scalarLiteral }* }``, as in "integer { 0, 1 } means an integer number whose value is 0 or 1". The script ends with ``end operator``, as the page's examples and the grammar do; its syntax line writes ``end define operator``.
:Expected: ``SC_r = 20``, an Integer: ``2`` is in the list, and ``2 * 10 = 20``.
