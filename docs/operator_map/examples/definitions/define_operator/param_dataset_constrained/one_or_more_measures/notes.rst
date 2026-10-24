:Tests: the multiplicity modifier ``_+``, ``dataset { measure<integer> _+ }``, with the two integer Measures of ``DS_1``.
:Manual: `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: ``multiplicityModifier ::= _ { + | * }``, and in the User Manual "The symbol ``_+`` means that the preceding type may appear from 1 to many times". The script ends with ``end operator``, as the page's examples do; its syntax line writes ``end define operator``.
:Expected: both Measures doubled: (1, 20, 2) and (2, 40, 4).
:Finding: both engines raise SemanticError ``1-3-13``, "Unsupported VTL construct: multiplicity modifier in 'measure<integer> _+'".
