:Tests: unary clauses only, ``when "N"`` before ``when "C"``, so a pair with N and C gives N.
:Manual: `define viral propagation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html>`__: a unary clause "matches if the specified value is present in the input pair (regardless of what the other value is)", and "The first matching clause determines the result".
:Expected: ``At_1`` N for (C, N), C for (C, F) and F for (F, F) by ``else``.
