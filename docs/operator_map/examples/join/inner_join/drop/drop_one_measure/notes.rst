:Tests: ``drop Me_1``.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: "drop will remove all specified non-identifier components of VDS₃".
:Expected: the natural join keeps ``(1,A)``, ``(2,A)`` and ``(2,B)``, the keys in both operands, with the identifiers and ``Me_2`` only: 1, 3 and 4.
