:Tests: ``random(SC_1, SC_2)`` on scalars, with ``SC_1`` = 15 and ``SC_2`` = 12 as in the page's valid syntax ``random(15,12)``.
:Manual: `Random <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Random/index.html>`__: "The operator returns a random decimal number >= 0 and <1."
:Expected: ``SC_r`` is a Number in ``[0, 1)`` that is the same on every run; a draw has no fixed value, so the expected value is null and the property is checked.
