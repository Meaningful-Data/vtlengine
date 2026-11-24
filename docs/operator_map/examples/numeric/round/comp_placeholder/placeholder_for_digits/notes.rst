:Tests: the placeholder ``_`` for ``numDigit``, on a component inside ``calc``.
:Manual: `Rounding <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Rounding/index.html>`__: "``round ( 12345.6, _ )`` gives ``12346``".
:Expected: ``Me_2 = round(Me_1, _)`` is an Integer, as with ``numDigit`` omitted: 3.14159, 2.71828 and 12345.6 give 3, 3 and 12346; the other components are kept.
