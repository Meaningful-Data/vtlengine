:Tests: ``ln`` of an integer.
:Manual: `Natural logarithm <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Natural%20logarithm/index.html>`__: "``ln ( 148 )`` gives ``4.997...``", and "The result has type number."
:Expected: ``SC_r = 4.997212`` (``ln 148 = ln 100 + ln 1.48 = 4.605170 + 0.392042``), a Number although the operand is an Integer.
