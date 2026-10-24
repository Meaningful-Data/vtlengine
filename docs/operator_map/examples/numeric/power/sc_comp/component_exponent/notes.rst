:Tests: ``power`` of a scalar base with the exponent in a component, inside ``calc``.
:Manual: `Power <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Power/index.html>`__ types ``exponent`` as ``component<number> | number``, so it may be a component while the base is a scalar.
:Expected: ``Me_2 = power(2, Me_1)``: ``2^3 = 8``, ``2^-1 = 0.5`` and ``2^10 = 1024``, a Number although both operands are Integers; the other components are kept.
