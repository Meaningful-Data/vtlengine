:Tests: a parameter of type ``string``, called as ``with_suffix(SC_1)`` with ``SC_1`` = ``A``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: a parameter type is "a VTL data type" (``basicScalarType ::= ... string ...``). The script ends with ``end operator``, as the page's examples and the grammar do; its syntax line writes ``end define operator``.
:Expected: ``SC_r = "A_B"``, a String: ``"A" || "_B"``.
