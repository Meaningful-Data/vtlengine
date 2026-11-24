:Tests: the Hamming distance between a component and a scalar, inside ``calc``.
:Manual: `String distances <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20distances/index.html>`__, Example 3, output verbatim: "Returns the Hamming distance, which is the number of positions at which the corresponding characters are different. Both strings must have the same length."
:Expected: ``Me_2`` is 1 for ``foo`` and ``fob``, which differ at the 3rd position only, and 3 for ``bar`` and ``fob``, which differ at all three; ``Me_1`` is kept.
