:Tests: ``aggr`` by ``Id_1`` with ``having min(Me_1) > 15``.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__, and Aggregate invocation: "only groups for which havingCondition evaluates to true appear in the result", and the condition "must invoke aggregate operators" (``min`` here, which the reading of #1119 also allows).
:Expected: the left join keeps the four keys of ``DS_1``, with ``Me_2`` null at ``(1,B)``. The smallest ``Me_1`` of ``Id_1`` = 1 is 10, so only ``Id_1`` = 2 is kept, with ``Me_3`` = 70.
