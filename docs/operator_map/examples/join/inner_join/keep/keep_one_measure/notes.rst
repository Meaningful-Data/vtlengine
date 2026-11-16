:Tests: ``keep Me_1``.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: "keep will maintain all the identifiers and only the specified non-identifier components of VDS₃".
:Expected: the natural join keeps ``(1,A)``, ``(2,A)`` and ``(2,B)``, the keys in both operands, with ``Me_1`` only: 10, 30 and 40.
