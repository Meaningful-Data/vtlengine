:Tests: ``aggr`` with no grouping clause.
:Manual: `Cross Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__: "If no grouping clause is specified, then all the input Data Points are aggregated in a single group and the clause returns a data set that contains a single Data Point and has no Identifier Components."
:Expected: the cartesian product of ``DS_1`` (``Id_1`` 1, 2) and ``DS_2`` (``Id_2`` A, B) has 2 × 2 = 4 data points, so one data point with no identifier: ``Me_3`` = 10 + 10 + 20 + 20 = 60.
