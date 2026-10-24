:Tests: the propagation at ``hierarchy(DS_1, HR_1 rule Id_2)``, with ``TOTAL = X + Y``.
:Manual: `Hierarchical roll-up <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html>`__: "The Attribute propagation rule is applied on the groups of Data Points which contribute to the same Data Points of the result", and the default output ``computed`` keeps "only the set of Data Points computed according to the Ruleset".
:Expected: one ``TOTAL`` per ``Id_1``: 10 with ``At_1`` C for (C, F), and 3 with F for (F, F).
