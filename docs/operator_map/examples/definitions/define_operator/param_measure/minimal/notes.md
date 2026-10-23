# define operator: a parameter of type `measure<integer>`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `c` with the component type `measure<integer>`
(`componentRole ::= ... | measure | ...`). The call passes the Measure `Me_1`.

The call is made inside `calc`, where the argument is a component of `DS_1` and the result a component (Clause operators, calc: "calcExpr :: component<scalar>"). `calc` keeps the components of `DS_1` and adds the new Measure `Me_2` ("If the calculated Component is a new Component, it is added to the output Data Set", and a new component without a role is a Measure).

Expected: `Me_2 = Me_1 * 2`, an integer: (1, 10, 20) and (2, 20, 40).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
