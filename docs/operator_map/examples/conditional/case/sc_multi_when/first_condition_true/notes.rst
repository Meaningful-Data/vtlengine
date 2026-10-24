:Tests: ``case when 3 > 0 then 2 when 3 = 0 then 0 else 5``: two ``when`` clauses, the first one true.
:Manual: `Case <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Case/index.html>`__: "case when x1 > x2 then 2 when x1 = x2 then 0 else 5; for x1 = 3, x2 = 0 it returns 2" (the literals 3 and 0 stand for x1 and x2).
:Expected: ``SC_r = 2``, an Integer: the first condition ``3 > 0`` is true, so the result is its thenOperand.
