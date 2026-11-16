:Tests: ``Me_1 > 20`` inside ``calc``, where the new measure ``Me_2`` takes the result.
:Manual: `Greater than <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Greater%20than/index.html>`__, Example 2, output verbatim: "The operator **>** returns **true** if *left* is greater than *right*, **false** otherwise."
:Expected: ``Me_2`` is true for F, false for R and null for G, whose ``Me_1`` is null; ``Me_1`` is kept.
