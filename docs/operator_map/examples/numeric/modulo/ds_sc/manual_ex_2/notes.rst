:Tests: every measure of ``DS_1`` modulo a scalar.
:Manual: `Modulo <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Modulo/index.html>`__, Example 2, output verbatim: "The operator **mod** returns the remainder of *op1* divided by *op2*."
:Expected: ``mod(DS_1, 15)``: ``100 mod 15 = 10``, ``18.45 mod 15 = 3.45``, ``20 mod 15 = 5`` and ``20.3 mod 15 = 5.3``; a value under 15 stays as it is.
