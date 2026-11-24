:Tests: ``drop Me_1``.
:Manual: `Cross Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__: "drop will remove all specified non-identifier components of VDS₃".
:Expected: the cartesian product of ``DS_1`` (``Id_1`` 1, 2) and ``DS_2`` (``Id_2`` A, B) has 2 × 2 = 4 data points, with the identifiers and ``Me_2`` only: 1, 2, 1 and 2.
