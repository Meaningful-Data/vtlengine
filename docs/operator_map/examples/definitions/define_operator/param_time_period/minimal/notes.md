# define operator: a parameter of type `time_period`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares the parameter `x` with the type `time_period` (in the Data Type
Syntax, `basicScalarType ::= ... time_period ...`). The call passes a value of that type, so `x` takes it inside the body.

Expected: `SC_r` = `2020Q1` (Time_Period), because the body returns the argument unchanged. A time_period has no literal, so the argument is built with `cast` from the manual's representation `2020Q1` (the first quarter of 2020).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
