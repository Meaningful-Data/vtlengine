:Tests: Nested parentheses: the inner ones are evaluated first.
:Manual: `Parentheses <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Parentheses/index.html>`__: "If more parentheses are nested, the inner parentheses are evaluated first, for example ``( 20 - 10 / (2 + 3) ) * 3`` would give ``54``."
:Expected: ``2 + 3 = 5``, ``10 / 5 = 2``, ``20 - 2 = 18`` and ``18 * 3 = 54``: ``SC_r = 54``, a Number, as a division gives a number (Division page) and so does an operation with a number operand (Subtraction and Multiplication pages).
