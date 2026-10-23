# define operator: a parameter of type `string`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares the parameter `x` with the type `string` (in the Data Type
Syntax, `basicScalarType ::= ... string ...`). The call passes a value of that type, so `x` takes it inside the body.

Expected: `SC_r` = `A_B` (String), because `"A" || "_B"` concatenates the two strings into `A_B`.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
