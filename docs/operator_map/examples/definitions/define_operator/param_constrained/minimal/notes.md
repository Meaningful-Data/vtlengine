# define operator: a parameter of type `integer [value >= 0]`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares the parameter `x` with the type `integer [value >= 0]` (in the Data Type
Syntax, `scalarType ::= basicScalarType { scalarTypeConstraint }, scalarTypeConstraint ::= [ valueBooleanCondition ] | ...`). The call passes a value of that type, so `x` takes it inside the body.

Expected: `SC_r` = `20` (Integer), because `2` satisfies the constraint (the manual: "the valueBooleanCondition in scalarTypeConstraint is expressed with reference to the fictitious variable value", as in `number [ value >= 0 ]`) and `2 * 10` = 20.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
