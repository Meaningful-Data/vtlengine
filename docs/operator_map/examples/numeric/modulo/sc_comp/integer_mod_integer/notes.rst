:Tests: the remainder of a scalar divided by an Integer component, inside ``calc``.
:Manual: `Modulo <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Modulo/index.html>`__: "If the type of both operands is *integer* then the result has type *integer*."
:Expected: ``Me_3 = mod(11, Me_1)``: ``11 mod 7 = 4``, ``11 mod 9 = 2`` and ``11 mod 10 = 1``, an Integer; the other components are kept.
