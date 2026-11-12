:Tests: ``DS_2 > DS_3``: the measures of two data sets, data point by data point.
:Manual: `Greater than <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Greater%20than/index.html>`__, Example 3, output verbatim: "The operator **>** returns **true** if *left* is greater than *right*, **false** otherwise." The page adds that "if the Me_1 column for Germany in the DS_2 or DS_3 Data Set had a **null** value the result would be null for Germany (G)".
:Expected: ``bool_var`` is false for G (7.1 against 7.5) and true for R (42.5 against 33.7).
