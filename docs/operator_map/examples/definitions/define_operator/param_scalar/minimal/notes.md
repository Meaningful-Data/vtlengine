# define operator: a parameter of type `scalar`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares the parameter `x` with the type `scalar` (in the Data Type
Syntax, `basicScalarType ::= scalar | ...`). The call passes a value of that type, so `x` takes it inside the body.

Expected: `SC_r` = `5` (Integer), because the body returns the argument unchanged. `scalar` is the super-type of all the scalar types (User Manual: "The super-type of all the scalar types is the type scalar, which means any scalar value"), so it takes the integer `5`, and the result keeps its type: `returns scalar` declares any scalar type.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
