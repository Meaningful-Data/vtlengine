:Tests: ``keep Me_1``.
:Manual: `Cross Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__: "keep will maintain all the identifiers and only the specified non-identifier components of VDS₃".
:Expected: the cartesian product of ``DS_1`` (``Id_1`` 1, 2) and ``DS_2`` (``Id_2`` A, B) has 2 × 2 = 4 data points, with ``Me_1`` only: 10, 10, 20 and 20.
