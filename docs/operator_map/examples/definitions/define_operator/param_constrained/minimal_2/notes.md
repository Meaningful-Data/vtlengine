# define operator: a parameter of type `integer { 1, 2, 3 }`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares the parameter `x` with the type `integer { 1, 2, 3 }` (in the Data Type
Syntax, `scalarTypeConstraint ::= ... | { scalarLiteral { , scalarLiteral }* }`). The call passes a value of that type, so `x` takes it inside the body.

Expected: `SC_r` = `20` (Integer), because `2` is one of the listed values (the manual: "integer { 0, 1 } means an integer number whose value is 0 or 1") and `2 * 10` = 20. This is the second form of scalarTypeConstraint, the list of allowed values.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
