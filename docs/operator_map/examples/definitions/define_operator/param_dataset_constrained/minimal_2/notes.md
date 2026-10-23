# define operator: a parameter of type `dataset { ... }` with a multiplicity modifier

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `ds` with a dataset type whose constraint uses the
multiplicity modifier `_` instead of a component name (`multiplicityModifier ::= _ { + | * }`).
The User Manual: "The underscore _ indicates that the preceding type appears just one time,
for example measure<string> _ indicates just one Measure having the scalar type string". So
`DS_1`, with one integer Measure, has this type. This is the same form the manual uses for the
operands of its own operators, e.g. `dataset { measure<number> _ }`.

Expected: `Me_1 * 2` for each data point: (1, 20) and (2, 40).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
