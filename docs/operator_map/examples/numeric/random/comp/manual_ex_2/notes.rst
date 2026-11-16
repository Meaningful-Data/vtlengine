:Tests: ``random`` on a component inside ``calc``, next to the seed ``Me_1``.
:Manual: `Random <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Random/index.html>`__, Example 2: "The operator returns a random decimal number >= 0 and <1." The page prints 0.7545341, 0.3457166 and 0.5183224 only to illustrate.
:Expected: ``Me_2`` holds numbers in ``[0, 1)`` that are the same on every run; the seed ``Me_1`` (16.0, 4.0 and 7.2) is kept and compared as written.
:Adaptation: no engine can reproduce the printed draws, so ``Me_2`` is blank in the expected file and is checked by that property.
