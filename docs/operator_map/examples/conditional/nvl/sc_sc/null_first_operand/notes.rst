:Tests: ``nvl ( null, SC_1 )`` with ``SC_1`` = 0: a null first operand on scalars.
:Manual: `Nvl <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Nvl/index.html>`__: "nvl ( null, 0 ) returns 0".
:Expected: ``SC_r = 0``, an Integer: op1 is null, so the result is op2.
