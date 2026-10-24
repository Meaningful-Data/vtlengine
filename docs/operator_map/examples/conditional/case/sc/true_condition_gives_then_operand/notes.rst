:Tests: ``case when 3 > 0 then 2 else 5``: one ``when``, with a true condition.
:Manual: `Case <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Case/index.html>`__: "The **case** operator returns the first *thenOperand* whose corresponding *condition* evaluates to **true**, *elseOperand* if none of the **when** conditions evaluates to **true**."
:Expected: ``SC_r = 2``, an Integer: ``3 > 0`` is true, so the result is its thenOperand.
