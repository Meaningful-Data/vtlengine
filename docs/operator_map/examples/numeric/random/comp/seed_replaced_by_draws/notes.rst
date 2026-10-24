:Tests: ``random`` whose result component is the seed measure itself.
:Manual: `Random <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Random/index.html>`__: "The operator returns a random decimal number >= 0 and <1."
:Expected: ``Me_1`` is replaced by the draws, as any operator on components replaces a measure of the same name; no seed column is left to compare, so the expected file has the structure only and the draws are checked as numbers in ``[0, 1)`` that are the same on every run.
:Finding: both engines pass the check, but they draw different values for the same seed.
