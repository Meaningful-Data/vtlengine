:Tests: ``union ( DS_1, DS_3 )``: two data sets that share one data point, (2012, B, Total, Total), with the measure 5 in ``DS_1`` and 23 in ``DS_3``.
:Manual: `Union <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Set%20operators/Union/index.html>`__, Example 2, output verbatim: duplicate Data Points "are filtered by choosing the Data Point belonging to the left most operand Data Set".
:Expected: the page's four data points: (2012, B, Total, Total) has the measure 5 of ``DS_1``, the leftmost operand, and not the 23 of ``DS_3``.
