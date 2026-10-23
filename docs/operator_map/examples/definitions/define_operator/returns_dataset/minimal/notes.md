# define operator: `returns` a dataset type

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `returns dataset` (`outputParameterType ::= ... |
datasetType`). The body adds 1 to the Measure of the operand (Typical behaviours: "the operator
is applied to the values of all the Measures").

Expected: `Me_1 + 1` for each data point: (1, 11) and (2, 21).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
