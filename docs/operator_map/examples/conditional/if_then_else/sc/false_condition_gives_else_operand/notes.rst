:Tests: ``if SC_1 > SC_2 then 2 else 5`` with ``SC_1`` = 0 and ``SC_2`` = 3: a false condition on scalars.
:Manual: `if-then-else <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/if-then-else/index.html>`__: "The **if** operator returns *thenOperand* if *condition* evaluates to **true**, *elseOperand* otherwise", for example "if x1 > x2 then 2 else 5, for x1 = 0, x2 =3 it returns 5" (``SC_1`` and ``SC_2`` stand for x1 and x2).
:Expected: ``SC_r = 5``, an Integer: ``SC_1 > SC_2`` (``0 > 3``) is false, so the result is the elseOperand.
