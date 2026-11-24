:Tests: ``aggr`` by ``Id_1`` with ``having min(Me_1) > 15``.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__, and Aggregate invocation: "only groups for which havingCondition evaluates to true appear in the result", and the condition "must invoke aggregate operators" (``min`` here, which the reading of #1119 also allows).
:Expected: the natural join keeps ``(1,A)``, ``(2,A)`` and ``(2,B)``, the keys in both operands. The smallest ``Me_1`` of ``Id_1`` = 1 is 10, so only ``Id_1`` = 2 is kept, with ``Me_3`` = 70.
