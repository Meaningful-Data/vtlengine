:Tests: a dataset parameter whose constraints name its components, ``dataset { identifier<integer> Id_1, measure<integer> Me_1 }``: the structure of ``DS_1``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``componentConstraint ::= componentType { componentName | multiplicityModifier }``. The script ends with ``end operator``, as the page's examples do; its syntax line writes ``end define operator``.
:Expected: ``Me_1 * 2`` on each data point: (1, 20) and (2, 40).
