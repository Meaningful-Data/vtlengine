:Tests: an operator that ``returns component<integer>``, called inside ``calc`` with the Measure ``Me_1``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``outputParameterType ::= scalarType | componentType | datasetType``. The script ends with ``end operator``, as the page's examples do; its syntax line writes ``end define operator``.
:Expected: ``Me_2 = Me_1 + 1``, a new Measure that ``calc`` adds: (1, 10, 11) and (2, 20, 21).
