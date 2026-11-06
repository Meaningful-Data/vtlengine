:Tests: ``if SC_1 > SC_2 then 2 else 5`` with ``SC_1`` = 3 and ``SC_2`` = 0: a true condition on scalars.
:Manual: `if-then-else <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/if-then-else/index.html>`__: "The **if** operator returns *thenOperand* if *condition* evaluates to **true**, *elseOperand* otherwise", for example "if x1 > x2 then 2 else 5, for x1 = 3, x2 =0 it returns 2" (``SC_1`` and ``SC_2`` stand for x1 and x2).
:Expected: ``SC_r = 2``, an Integer: ``SC_1 > SC_2`` (``3 > 0``) is true, so the result is the thenOperand.
