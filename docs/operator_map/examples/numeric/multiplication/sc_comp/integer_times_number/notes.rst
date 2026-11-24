:Tests: an integer scalar multiplied by a Number component, inside ``calc``.
:Manual: `Multiplication <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Multiplication/index.html>`__: "If one of the operands is of type *number*, then the other operand is implicitly cast to *number* and therefore the result has type *number*."
:Expected: ``Me_3 = 3 * Me_2``: ``3 * 2.5 = 7.5``, ``3 * 4.0 = 12.0`` and ``3 * 0.5 = 1.5``, a Number; the other components are kept.
