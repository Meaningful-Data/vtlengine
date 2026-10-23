# define operator: a parameter of type `dataset`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `ds` with the dataset type `dataset` (`datasetType ::=
dataset { ... }`, here without component constraints). The call passes `DS_1`, and the body
multiplies its Measure (Typical behaviours: "the operator is applied to the values of all the
Measures").

Expected: `Me_1 * 2` for each data point: (1, 20) and (2, 40).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
