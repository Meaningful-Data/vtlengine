:Tests: a parameter of type ``integer``, called as ``add_one(SC_1)`` with ``SC_1`` = 2.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: a parameter type is "a VTL data type" (``basicScalarType ::= ... integer ...``). The script ends with ``end operator``, as the page's examples and the grammar do; its syntax line writes ``end define operator``.
:Expected: ``SC_r = 3``, an Integer: ``2 + 1``.
