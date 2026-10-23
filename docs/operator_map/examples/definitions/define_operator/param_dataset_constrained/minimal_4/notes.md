# define operator: a parameter of type `dataset { measure<integer> _* }`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `ds` with a dataset type whose constraint uses the
multiplicity modifier `_*` (`multiplicityModifier ::= _ { + | * }`). The User Manual:
"The symbol _* means that the preceding type may appear from 0 to many times". `DS_1` has one integer Measure, and the body doubles it: (1, 20) and (2, 40).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
