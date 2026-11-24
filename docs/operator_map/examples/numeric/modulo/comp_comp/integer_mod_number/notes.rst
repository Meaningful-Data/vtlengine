:Tests: the remainder of an Integer component divided by a Number component, inside ``calc``.
:Manual: `Modulo <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Modulo/index.html>`__: "If one of the operands is of type *number*, then the other operand is implicitly cast to *number* and therefore the result has type *number*."
:Expected: ``Me_3 = mod(Me_1, Me_2)``: ``7 mod 2.5 = 2.0``, ``9 mod 4.0 = 1.0`` and ``10 mod 3.5 = 3.0``, a Number; the other components are kept.
