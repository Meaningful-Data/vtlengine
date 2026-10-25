:Tests: ``setdiff ( DS_3, DS_4 )`` on data sets with three identifiers, where ``DS_4`` holds two of the three data points of ``DS_3``.
:Manual: `Set difference <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Set%20operators/Set%20difference/index.html>`__, Example 2, output verbatim: the resulting Dataset contains "all the data points Data Point *dp1* of the operand *ds1*, such that there is no Data Point *dp2* of *ds2* having the same values for homonym Identifier Components".
:Expected: the page's one data point, (R, T, 2011) with the measure 12.
