:Tests: ``case when SC_1 > SC_2 then 2 when SC_1 = SC_2 then 0 else 5`` with ``SC_1`` = 3 and ``SC_2`` = 0: two ``when`` clauses, the first one true.
:Manual: `Case <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Case/index.html>`__: "case when x1 > x2 then 2 when x1 = x2 then 0 else 5; for x1 = 3, x2 = 0 it returns 2" (``SC_1`` and ``SC_2`` stand for x1 and x2).
:Expected: ``SC_r = 2``, an Integer: the first condition ``SC_1 > SC_2``, that is ``3 > 0``, is true, so the result is its thenOperand.
