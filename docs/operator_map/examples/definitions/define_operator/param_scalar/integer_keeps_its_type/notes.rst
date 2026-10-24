:Tests: a parameter of type ``scalar``, with ``returns scalar``, called with the integer ``5``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``basicScalarType ::= scalar | ...``, and in the User Manual "The super-type of all the scalar types is the type scalar". The script ends with ``end operator``, as the page's examples do; its syntax line writes ``end define operator``.
:Expected: ``SC_r = 5``, an Integer: the body returns its argument, which keeps its type, as ``returns scalar`` allows any scalar type.
