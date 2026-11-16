:Tests: ``case when SC_1 > SC_2 then 2 else 5`` with ``SC_1`` = 3 and ``SC_2`` = 0: one ``when``, with a true condition.
:Manual: `Case <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Case/index.html>`__: "The **case** operator returns the first *thenOperand* whose corresponding *condition* evaluates to **true**, *elseOperand* if none of the **when** conditions evaluates to **true**."
:Expected: ``SC_r = 2``, an Integer: ``SC_1 > SC_2``, that is ``3 > 0``, is true, so the result is its thenOperand.
