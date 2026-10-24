:Tests: a parameter of type ``integer not null``, called with ``2``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``scalarType ::= ... { { not } null }``. The script ends with ``end operator``, as the page's examples do; its syntax line writes ``end define operator``.
:Expected: ``SC_r = 3``, an Integer: ``2`` is not null, and ``2 + 1 = 3``.
:Finding: both engines raise SemanticError ``1-3-13``, "Unsupported VTL construct: not null in 'integer not null'".
