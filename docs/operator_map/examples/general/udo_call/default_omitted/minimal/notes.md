# Call that omits a trailing optional argument

Authored. [User-defined operator call](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/User-defined%20operator%20call/index.html): "One or more optional operands in the last
positions can be simply omitted", and the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) page: "If
parameterDefaultValue is specified then the parameter is optional". `y` is omitted and takes its
default 1, so `x - y` = 10 - 1 = 9, an integer.
