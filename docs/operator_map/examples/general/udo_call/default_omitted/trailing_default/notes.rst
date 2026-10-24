:Tests: A call that omits its last argument, an optional one: ``diff2(10)``.
:Manual: `User-defined operator call <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/User-defined%20operator%20call/index.html>`__: "One or more optional operands in the last positions can be simply omitted", and `define operator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__: "If parameterDefaultValue is specified then the parameter is optional".
:Expected: ``y`` takes its default 1, so ``SC_r = 10 - 1 = 9``, an Integer.
