:Tests: ``numDigit`` given by an Integer component, which may be negative.
:Manual: `Rounding <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Rounding/index.html>`__ types ``numDigit`` as ``component<integer> | integer``, and "``round ( 12345.6, -1 )`` gives ``12350.0``".
:Expected: ``Me_3 = round(Me_1, Me_2)``: ``round(3.14159, 2) = 3.14``, ``round(2.71828, 3) = 2.718`` and ``round(12345.6, -1) = 12350.0``, a Number; the other components are kept.
