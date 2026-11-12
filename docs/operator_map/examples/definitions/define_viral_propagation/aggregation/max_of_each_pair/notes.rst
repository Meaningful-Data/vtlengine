:Tests: ``aggregate max`` on the Number viral ``At_1``, at ``DS_1 + DS_2``.
:Manual: `define viral propagation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html>`__: "The rule simply names the aggregate function that must collapse the set of operand values to a single result", and "The semantics of these aggregates follow the standard VTL definitions".
:Expected: ``At_1`` 4.0 for (1.5, 4.0) and 3.0 for (3.0, 2.0); ``Me_1`` 11 and 22.
