:Tests: ``between(10, 5, 10)``: an operand equal to the upper limit.
:Manual: `Between <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Between/index.html>`__: "The operator returns **true** if *op* is greater than or equal to *from* and lower than or equal to *to*. In other terms, it is a shortcut for the following: ``op >= from and op <= to``".
:Expected: ``SC_r = true``, a Boolean: 10 >= 5 and 10 <= 10; it would be false if the limit were excluded.
