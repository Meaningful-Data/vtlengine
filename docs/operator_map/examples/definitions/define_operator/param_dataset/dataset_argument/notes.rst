:Tests: a parameter of type ``dataset``, with no component constraint, called with ``DS_1``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``datasetType ::= dataset { ... }``, here without constraints: the super-type of the Data Set types, "any dataset" (User Manual). The script ends with ``end operator``, as the page's examples do; its syntax line writes ``end define operator``.
:Expected: ``Me_1 * 2`` on each data point, as "the operator is applied to the values of all the Measures": (1, 20) and (2, 40).
