:Tests: aliases on the operands (``as d1``, ``as d2``), with no clause.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: the aliases are "valid only within the “join” operation to make it easier to refer to them".
:Expected: the aliases do not change the result: that of Common keys only, ``(1,A)``, ``(2,A)`` and ``(2,B)`` with ``Me_1`` and ``Me_2``.
