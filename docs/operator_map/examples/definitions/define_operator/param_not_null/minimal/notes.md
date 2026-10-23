# define operator: a parameter of type `integer not null`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares the parameter `x` with the type `integer not null` (in the Data Type
Syntax, `scalarType ::= ... { { not } null }`). The call passes a value of that type, so `x` takes it inside the body.

Expected: `sc_r` = `3` (Integer), because `2` is not null and `2 + 1` = 3.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
