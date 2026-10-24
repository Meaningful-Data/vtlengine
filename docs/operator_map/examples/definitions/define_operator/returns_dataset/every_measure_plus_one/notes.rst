:Tests: an operator that ``returns dataset``, called with ``DS_1``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``outputParameterType ::= scalarType | componentType | datasetType``. The script ends with ``end operator``, as the page's examples do; its syntax line writes ``end define operator``.
:Expected: ``Me_1 + 1`` on each data point, as "the operator is applied to the values of all the Measures": (1, 11) and (2, 21).
