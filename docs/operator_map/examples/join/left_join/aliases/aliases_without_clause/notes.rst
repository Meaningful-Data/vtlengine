:Tests: aliases on the operands (``as d1``, ``as d2``), with no clause.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__: the aliases are "valid only within the “join” operation to make it easier to refer to them".
:Expected: the aliases do not change the result: that of Left keys kept, the four keys of ``DS_1`` with ``Me_2`` null at ``(1,B)``.
