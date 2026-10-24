:Tests: ``case when 0 > 3 then 2 else 5``: one ``when``, with a false condition.
:Manual: `Case <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Case/index.html>`__: "The **case** operator returns the first *thenOperand* whose corresponding *condition* evaluates to **true**, *elseOperand* if none of the **when** conditions evaluates to **true**."
:Expected: ``SC_r = 5``, an Integer: ``0 > 3`` is false and there is no other ``when``, so the result is the elseOperand.
