:Tests: ``DS_1 = 0.08``: every measure value of ``DS_1`` against a scalar, with a null measure.
:Manual: `Equal to <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Equal%20to/index.html>`__, Example 1, output verbatim: "The operator returns **true** if the left is equal to right, **false** otherwise."
:Expected: ``bool_var`` is true for F and W, whose measure is 0.08, false for G, S and M, and null for B, whose measure is null.
