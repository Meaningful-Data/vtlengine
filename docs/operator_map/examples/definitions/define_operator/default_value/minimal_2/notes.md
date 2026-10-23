# define operator: a parameter with a default value, passed as `_`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html): "If parameterDefaultValue is specified then the parameter is
optional". The [call](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/User-defined%20operator%20call/index.html) writes `_` for it: "An underscore (_) can be used to denote that
the value for an optional operand is omitted".

Expected: `y` takes its default 10, so `SC_r` = 1 + 10 = 11 (Integer).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
