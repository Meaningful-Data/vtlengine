:Tests: the multiplicity modifier ``_*``, ``dataset { measure<integer> _* }``, with the one integer Measure of ``DS_1``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``multiplicityModifier ::= _ { + | * }``, and in the User Manual "The symbol ``_*`` means that the preceding type may appear from 0 to many times". The script ends with ``end operator``, as the page's examples and the grammar do; its syntax line writes ``end define operator``.
:Expected: ``Me_1 * 2`` on each data point: (1, 20) and (2, 40).
:Finding: both engines raise SemanticError ``1-3-13``, "Unsupported VTL construct: multiplicity modifier in 'measure<integer> _*'".
