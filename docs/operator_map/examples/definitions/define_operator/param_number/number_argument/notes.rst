:Tests: a parameter of type ``number``, called as ``twice(1.25)``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: a parameter type is "a VTL data type" (``basicScalarType ::= ... number ...``). The script ends with ``end operator``, as the page's examples do; its syntax line writes ``end define operator``.
:Expected: ``SC_r = 2.5``, a Number: ``1.25 * 2``.
