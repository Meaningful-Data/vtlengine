:Tests: the sign of the remainder when the divisor is negative: ``mod(SC_1, SC_2)`` with ``SC_1`` = 5 and ``SC_2`` = -2.
:Manual: `Modulo <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Modulo/index.html>`__: "``mod ( 5, -2 )`` gives ``-1``".
:Expected: ``SC_r = -1``, an Integer: the remainder takes the sign of the divisor.
:Finding: both engines return ``1``, not ``-1``.
