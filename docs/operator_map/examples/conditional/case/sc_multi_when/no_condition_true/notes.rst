:Tests: ``case when 0 > 3 then 2 when 0 = 3 then 0 else 5``: no condition true.
:Manual: `Case <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Case/index.html>`__: "case when x1 > x2 then 2 when x1 = x2 then 0 else 5; for x1 = 0, x2 = 3 it returns 5" (the literals 0 and 3 stand for x1 and x2).
:Expected: ``SC_r = 5``, an Integer: neither ``0 > 3`` nor ``0 = 3`` is true, so the result is the elseOperand.
