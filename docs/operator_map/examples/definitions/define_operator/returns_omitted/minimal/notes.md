# define operator: no `returns`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html): `{ returns outputType }` is optional, and "If outputType is omitted
then the type returned by the operatorBody expression is assumed".

Expected: the body `x + 1` returns an integer, so `sc_r` = 2 + 1 = 3 (Integer).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
