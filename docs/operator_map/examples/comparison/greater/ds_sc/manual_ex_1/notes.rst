:Tests: ``DS_1 > 20``: every measure value of ``DS_1`` against a scalar, with a null measure.
:Manual: `Greater than <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Greater%20than/index.html>`__, Example 1, output verbatim: "The operator **>** returns **true** if *left* is greater than *right*, **false** otherwise."
:Expected: ``bool_var`` is true for F (29.5), false for R (12.2) and null for G, whose measure is null.
