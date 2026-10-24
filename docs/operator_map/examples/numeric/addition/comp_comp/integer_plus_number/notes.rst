:Tests: the sum of an Integer component and a Number component, inside ``calc``.
:Manual: `Addition <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Addition/index.html>`__: "If one of the operands is of type *number*, then the other operand is implicitly cast to *number* and therefore the result has type *number*."
:Expected: ``Me_3 = Me_1 + Me_2``: ``2 + 2.5 = 4.5``, ``4 + 4.0 = 8.0`` and ``5 + 0.5 = 5.5``, a Number; the other components are kept.
