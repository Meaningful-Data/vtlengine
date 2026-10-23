# define operator: a parameter of type `dataset { measure<integer> _+ }`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `ds` with a dataset type whose constraint uses the
multiplicity modifier `_+` (`multiplicityModifier ::= _ { + | * }`). The User Manual:
"The symbol _+ means that the preceding type may appear from 1 to many times, for example measure<string> _+ means one or more generic Measures having the scalar type string". `DS_1` has two integer Measures, `Me_1` and `Me_2`, and the body doubles both: (1, 20, 2) and (2, 40, 4).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
