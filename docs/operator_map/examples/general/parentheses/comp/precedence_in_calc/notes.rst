:Tests: Parentheses on components inside ``calc``: ``(Me_1 + 2) * 3`` adds before it multiplies.
:Manual: `Parentheses <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Parentheses/index.html>`__: "The operations enclosed in the parentheses are evaluated first", with ``( CMP_1 - CMP_2 )`` among the valid syntaxes and ``op`` of type ``dataset | component | scalar``.
:Expected: ``Me_2`` is ``(1 + 2) * 3 = 9`` and ``(4 + 2) * 3 = 18``, Integers; without the parentheses the multiplication would come first and give 7 and 10.
