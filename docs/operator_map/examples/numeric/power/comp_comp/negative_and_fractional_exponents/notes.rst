:Tests: ``power`` with the exponent in a Number component, inside ``calc``.
:Manual: `Power <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Power/index.html>`__ types ``exponent`` as ``component<number> | number``, and "``power ( 5, -1 )`` gives ``0.2``".
:Expected: ``Me_3 = power(Me_1, Me_2)``: ``2^3.0 = 8.0``, ``5^-1.0 = 0.2`` and ``4^0.5 = 2.0``, a Number; the other components are kept.
