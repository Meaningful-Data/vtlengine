# define operator: a parameter of type `duration`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares the parameter `x` with the type `duration` (in the Data Type
Syntax, `basicScalarType ::= ... duration ...`). The call passes a value of that type, so `x` takes it inside the body.

Expected: `SC_r` = `M` (Duration), because the body returns the argument unchanged. A duration has no literal, so the argument is built with `cast` from the manual's one-character code `M` (a month, User Manual, the codes of the duration type).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
