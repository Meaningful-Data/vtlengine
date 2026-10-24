:Tests: a parameter of type ``measure<integer>``, called inside ``calc`` with the Measure ``Me_1``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``componentRole ::= component | identifier | measure | attribute | viral attribute``. The script ends with ``end operator``, as the page's examples and the grammar do; its syntax line writes ``end define operator``.
:Expected: ``Me_2 = Me_1 * 2``, a new Measure that ``calc`` adds: (1, 10, 20) and (2, 20, 40).
