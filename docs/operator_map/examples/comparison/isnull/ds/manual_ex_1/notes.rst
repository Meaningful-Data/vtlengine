:Tests: ``isnull(DS_1)``: the measure of ``DS_1``, with two null values.
:Manual: `Is null <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Is%20null/index.html>`__, Example 1, output verbatim: "The operator returns **true** if the value of the operand is **null**, **false** otherwise."
:Expected: ``bool_var`` is true for S and W, whose measure is null, and false for B, G, M and F.
