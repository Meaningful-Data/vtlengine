# define operator: a parameter of type `number`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares the parameter `x` with the type `number` (in the Data Type
Syntax, `basicScalarType ::= ... number ...`). The call passes a value of that type, so `x` takes it inside the body.

Expected: `SC_r` = `2.5` (Number), because `1.25 * 2` = 2.5, a number.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
