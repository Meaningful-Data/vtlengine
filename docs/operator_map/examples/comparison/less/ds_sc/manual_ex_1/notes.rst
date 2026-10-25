:Tests: ``DS_1 < 15000000``: every measure value of ``DS_1`` against a scalar, with a null measure.
:Manual: `Less than <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Less%20than/index.html>`__, Example 1, output verbatim: "The operator **<** returns **true** if left is smaller than right, **false** otherwise."
:Expected: ``bool_var`` is true for B, G, F and W, false for S (46818219) and null for M, whose measure is null.
