:Tests: the named clauses ``R_C`` and ``R_N`` and an unnamed ``else``: the page's ``CONF_priority`` with names.
:Manual: `define viral propagation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html>`__: ``enumeratedClause ::= { ruleName : } when enumerationCondition then vpResult`` and ``defaultClause ::= else vpResult``.
:Expected: the names take no part in the matching: ``At_1`` C for (C, N), N for (N, F) and F for (F, F), as without names.
