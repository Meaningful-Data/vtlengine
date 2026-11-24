:Tests: ``aggr`` with no grouping clause.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__: "If no grouping clause is specified, then all the input Data Points are aggregated in a single group and the clause returns a data set that contains a single Data Point and has no Identifier Components."
:Expected: one data point with no identifier: ``Me_3`` = 10 + 20 + 30 + 40 = 100, the ``Me_1`` of the four keys of ``DS_1``.
