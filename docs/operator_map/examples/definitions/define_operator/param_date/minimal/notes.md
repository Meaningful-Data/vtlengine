# define operator: a parameter of type `date`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares the parameter `x` with the type `date` (in the Data Type
Syntax, `basicScalarType ::= ... date ...`). The call passes a value of that type, so `x` takes it inside the body.

Expected: `sc_r` = `2020-01-15` (Date), because the body returns the argument unchanged. The manual's types have no literals of their own (User Manual, Conventions: "the types time, date, time_period and duration do not directly support literals ... converted in the desired type by the cast operator"), so the argument is built with `cast`.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
