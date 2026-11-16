:Tests: aliases on the operands (``as d1``, ``as d2``), with no clause.
:Manual: `Cross Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__: the aliases are "valid only within the “join” operation to make it easier to refer to them".
:Expected: the aliases do not change the result: that of Cartesian product, the four pairs ``(1,A)``, ``(1,B)``, ``(2,A)`` and ``(2,B)``.
