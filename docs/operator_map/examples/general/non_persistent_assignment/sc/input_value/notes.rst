:Tests: ``SC_r := SC_1`` with the input scalar ``SC_1`` = 3: a scalar read from the input, assigned to a non-persistent result.
:Manual: `Non-persistent assignment <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Non-persistent%20assignment/index.html>`__: "If the Operand ``op`` is a scalar value, the result Data Set has no Components and contains only such a scalar value. For example, ``income := 3`` assigns the value ``3`` to the non-persistent Data Set named ``income``."
:Expected: ``SC_r = 3``, an Integer: the value of ``SC_1``.
