:Tests: two propagations, ``VP_1`` and ``VP_2``, on the same variable ``At_1``, before ``DS_1 + DS_2``.
:Manual: `define viral propagation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html>`__: "It is a semantic error to define more than one propagation rule for the same Value Domain or for the same Variable".
:Expected: the SemanticError ``1-3-3-1``, "Duplicate viral propagation rule for variable {name}.", as the page says: "If more than one propagation definition is declared for the same Variable, a semantic error is raised".
