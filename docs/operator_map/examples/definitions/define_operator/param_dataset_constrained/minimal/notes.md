# define operator: a parameter of type `dataset { ... }` with named components

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `ds` with a dataset type with component constraints
(`datasetType ::= dataset { componentConstraint { , componentConstraint }* }`,
`componentConstraint ::= componentType { componentName | multiplicityModifier }`). Here each
constraint names its component: an integer Identifier `Id_1` and an integer Measure `Me_1`,
which is the structure of `DS_1`.

Expected: `Me_1 * 2` for each data point: (1, 20) and (2, 40).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
