:Tests: a parameter of type ``identifier<integer>``, called inside ``calc`` with the Identifier ``Id_1``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``componentRole ::= component | identifier | measure | attribute | viral attribute``. The script ends with ``end operator``, as the page's examples do; its syntax line writes ``end define operator``.
:Expected: ``Me_2 = Id_1 * 10``, a new Measure that ``calc`` adds: (1, 10, 10) and (2, 20, 20).
