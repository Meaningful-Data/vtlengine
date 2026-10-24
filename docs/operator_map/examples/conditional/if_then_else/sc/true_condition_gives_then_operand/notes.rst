:Tests: ``if 3 > 0 then 2 else 5``: a true condition on scalars.
:Manual: `if-then-else <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/if-then-else/index.html>`__: "The **if** operator returns *thenOperand* if *condition* evaluates to **true**, *elseOperand* otherwise", for example "if x1 > x2 then 2 else 5, for x1 = 3, x2 =0 it returns 2" (the literals 3 and 0 stand for x1 and x2).
:Expected: ``SC_r = 2``, an Integer: ``3 > 0`` is true, so the result is the thenOperand.
