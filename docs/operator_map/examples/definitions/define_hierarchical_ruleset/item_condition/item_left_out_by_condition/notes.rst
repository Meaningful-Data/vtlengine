:Tests: ``A = B + C [Id_1 = "Y"]`` on the group ``X``: the right condition leaves ``C`` out.
:Manual: `define hierarchical ruleset <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/hierarchical_rulesets.html>`__: "the right member Code Item is taken into account only when the relevant rightCondition is true".
:Expected: ``A`` against ``B`` alone, 5 = 5: ``bool_var`` true and ``imbalance`` 0. Counting ``C`` would give false with ``imbalance`` 5 - 8 = -3.
