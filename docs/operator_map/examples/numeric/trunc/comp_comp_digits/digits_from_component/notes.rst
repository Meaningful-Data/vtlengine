:Tests: ``numDigit`` given by an Integer component, which may be negative.
:Manual: `Truncation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Truncation/index.html>`__ types ``numDigit`` as ``component<integer> | integer``, and "``trunc( 12345.6, -1 )`` gives ``12340.0``".
:Expected: ``Me_3 = trunc(Me_1, Me_2)``: ``trunc(3.14159, 2) = 3.14``, ``trunc(2.71828, 3) = 2.718`` and ``trunc(12345.6, -1) = 12340.0``, a Number; the other components are kept.
