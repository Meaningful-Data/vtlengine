:Tests: a propagation on the variable ``At_1``, the page's ``CONF_priority`` ("the CL_CONF list with C > N > F"), at ``DS_1 + DS_2``.
:Manual: `define viral propagation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html>`__: the variable is "the Represented Variable owning the viral Attribute", and a unary clause "matches if the specified value is present in the input pair (regardless of what the other value is)".
:Expected: ``At_1`` C for the pair (C, N), N for (N, F) and F for (F, F) by ``else``, and it stays viral.
