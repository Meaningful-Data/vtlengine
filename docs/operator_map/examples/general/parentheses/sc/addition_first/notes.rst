:Tests: Parentheses on scalars: ``(SC_1 + SC_2) * SC_3`` with ``SC_1`` = 2, ``SC_2`` = 3 and ``SC_3`` = 4 adds before it multiplies.
:Manual: `Parentheses <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Parentheses/index.html>`__: "For example ``(2+3)*4`` returns ``20``, instead ``2+3*4`` returns ``14`` because the multiplication has higher precedence than the addition."
:Expected: ``SC_r = 20``, an Integer, as the operands are integers.
