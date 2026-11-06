:Tests: ``Me_1 <> 7.5`` inside ``calc``, where the new measure ``Me_2`` takes the result.
:Manual: `Not equal to <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Not%20equal%20to/index.html>`__, Example 2, output verbatim: "The operator returns **false** if the left is equal to right, **true** otherwise."
:Expected: ``Me_2`` is true for G, where 7.1 differs from 7.5, and null for R, whose ``Me_1`` is null; ``Me_1`` is kept.
