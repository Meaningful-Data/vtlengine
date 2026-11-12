:Tests: ``if Me_1 > 10 then 1 else 0`` inside ``calc``: scalar branches, the usual way to build a flag.
:Manual: `if-then-else <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/if-then-else/index.html>`__: "If one of the operands *thenOperand* or *elseOperand* is scalar, such a scalar value can be returned depending on the outcome of the *condition*." Here both are.
:Expected: ``Me_2`` is 0, 1 and 0: only 15 is above 10, and 10 itself is not.
