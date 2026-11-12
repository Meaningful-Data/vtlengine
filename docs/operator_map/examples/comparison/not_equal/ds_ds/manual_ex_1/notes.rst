:Tests: ``DS_1 <> DS_2``: the measures of two data sets, data point by data point, with a null measure in ``DS_1``.
:Manual: `Not equal to <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Not%20equal%20to/index.html>`__, Example 1, output verbatim: "The operator returns **false** if the left is equal to right, **true** otherwise."
:Expected: ``bool_var`` is true for G, where 7.1 differs from 7.5, and null for R, whose measure in ``DS_1`` is null.
