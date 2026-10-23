# define operator: a parameter of type `attribute<string>`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `c` with the component type `attribute<string>`
(`componentRole ::= ... | attribute | ...`). The call passes the Attribute `At_1`.

The call is made inside `calc` and overwrites `At_1` itself (Clause operators, calc: "If the Calculated component is a Measure or an Attribute that already exists in the input Data Set, the calculated values overwrite the original values"). Without a role, "if the component exists in the operand Data Set then it maintains its role". Calculating the attribute explicitly also keeps the example away from the question of whether an attribute that is not calculated stays in the result: the attribute propagation rule "does not apply if the operations on the Attributes to be propagated are explicitly specified in the expression".

Expected: `At_1` stays an Attribute with `At_1 || "_X"`: (1, 10, A_X) and (2, 20, B_X).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
