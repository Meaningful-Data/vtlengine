:Tests: A call with ``_`` for an optional argument in the middle, ``diff3(10, _, 5)``, where it cannot simply be left out because ``z`` comes after it.
:Manual: `User-defined operator call <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/User-defined%20operator%20call/index.html>`__: "An underscore (“_”) can be used to denote that the value for an optional operand is omitted."
:Expected: ``x = 10``, ``y`` takes its default 1 and ``z = 5``, so ``SC_r = 10 - 1 - 5 = 4``, an Integer; reading 5 as ``y``, with ``z`` at its default 100, would give -95.
