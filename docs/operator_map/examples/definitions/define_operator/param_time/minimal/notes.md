# define operator: a parameter of type `time`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares the parameter `x` with the type `time` (in the Data Type
Syntax, `basicScalarType ::= ... time ...`). The call passes a value of that type, so `x` takes it inside the body.

Expected: `sc_r` = `2020-01-01/2020-12-31` (Time), because the body returns the argument unchanged. A time value has no literal, so the argument is built with `cast` from the manual's representation `YYYY-MM-DD/YYYY-MM-DD` (here the whole year 2020).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
