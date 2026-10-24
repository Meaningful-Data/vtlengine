:Tests: a propagation on the value domain ``CL_CONF``, the page's ``CONF_priority``, for ``At_1``, whose input structure records that it takes its values from ``CL_CONF``, at ``DS_1 + DS_2``.
:Manual: `define viral propagation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html>`__: "When a propagation rule is declared on a Value Domain it acts as the default for all Attributes using that Value Domain".
:Expected: ``At_1`` C for the pair (C, N), N for (N, F) and F for (F, F) by ``else``, and it stays viral.
:Finding: both engines raise SemanticError ``1-3-3-8``: a propagation "defined on value domain CL_CONF ... is not supported yet". Besides, vtlengine's loader ignores the ``variables`` and ``domains`` of its structure schema.
