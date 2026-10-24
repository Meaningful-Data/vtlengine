:Tests: ``case when 3 > 3 then 2 when 3 = 3 then 0 else 5``: the first condition false and the second one true.
:Manual: `Case <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Case/index.html>`__: "case when x1 > x2 then 2 when x1 = x2 then 0 else 5; for x1 = x2 = 3 it returns 0" (the literal 3 stands for x1 and x2).
:Expected: ``SC_r = 0``, an Integer: ``3 > 3`` is false and ``3 = 3`` is true, so the result is the thenOperand of the second ``when``.
