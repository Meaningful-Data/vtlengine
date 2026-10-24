:Tests: ``random(DS_1, 5)`` on a dataset: the measure is replaced by draws.
:Manual: `Random <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Random/index.html>`__, Example 1: "The operator returns a random decimal number >= 0 and <1." The page prints 0.3582791, 0.428819 and 0.715488 only to illustrate.
:Expected: the identifiers are the page's; ``Me_1`` holds numbers in ``[0, 1)`` that are the same on every run.
:Finding: both engines pass the check, but they draw different values for the same seed.
:Adaptation: no engine can reproduce the printed draws, so ``Me_1`` is blank in the expected file and the example is checked by that property.
