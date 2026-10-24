:Tests: ``random(15, 12)``, one of the page's valid syntaxes, on scalars.
:Manual: `Random <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Random/index.html>`__: "The operator returns a random decimal number >= 0 and <1."
:Expected: ``SC_r`` is a Number in ``[0, 1)`` that is the same on every run; a draw has no fixed value, so the expected value is null and the property is checked.
:Finding: both engines pass the check, but they draw different values for the same seed.
