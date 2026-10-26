:Tests: the result when the divisor is 0: ``mod(SC_1, SC_2)`` with ``SC_1`` = 9 and ``SC_2`` = 0.
:Manual: `Modulo <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Modulo/index.html>`__: "It returns *op1* if divisor *op2* is 0", and "``mod ( 9, 0 )`` gives ``9``".
:Expected: ``SC_r = 9``, an Integer.
:Finding: both engines refuse with RunTimeError ``2-1-15-6``, "Scalar division by Zero". That is the rule of the Division page ("If *op2* is 0 then the operation generates a run-time error"), not of Modulo.
