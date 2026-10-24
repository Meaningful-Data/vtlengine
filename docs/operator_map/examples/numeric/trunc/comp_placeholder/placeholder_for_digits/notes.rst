:Tests: the placeholder ``_`` for ``numDigit``, on a component inside ``calc``.
:Manual: `Truncation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Truncation/index.html>`__: "``trunc ( 12345.6, _ )`` gives ``12345``".
:Expected: ``Me_2 = trunc(Me_1, _)`` is an Integer, as with ``numDigit`` omitted: 3.14159, 2.71828 and 12345.6 give 3, 2 and 12345; the other components are kept.
