:Tests: ``case when SC_1 > SC_2 then 2 when SC_1 = SC_2 then 0 else 5`` with ``SC_1`` = 0 and ``SC_2`` = 3: no condition true.
:Manual: `Case <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Case/index.html>`__: "case when x1 > x2 then 2 when x1 = x2 then 0 else 5; for x1 = 0, x2 = 3 it returns 5" (``SC_1`` and ``SC_2`` stand for x1 and x2).
:Expected: ``SC_r = 5``, an Integer: neither ``SC_1 > SC_2`` (``0 > 3``) nor ``SC_1 = SC_2`` (``0 = 3``) is true, so the result is the elseOperand.
