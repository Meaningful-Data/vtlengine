:Tests: an operator with no ``returns``, ``add_one (x integer) is x + 1``, called as ``add_one(SC_1)`` with ``SC_1`` = 2.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: "If outputType is omitted then the type returned by the operatorBody expression is assumed". The script ends with ``end operator``, as the page's examples and the grammar do; its syntax line writes ``end define operator``.
:Expected: ``SC_r = 3``, an Integer, the type of ``x + 1``.
:Finding: both engines raise SemanticError ``1-3-2-2``, "User defined operator without returns is not implemented".
