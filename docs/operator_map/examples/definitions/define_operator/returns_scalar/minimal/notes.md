# define operator: `returns` a scalar type

The operator of Example 1 of [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html), with `returns integer`. The manual
writes `returns boolean` there, but `if x > y then x else y` returns one of the two integer
parameters, and "if outputType is specified then the type of operatorBody must be compatible
with outputType". So the example declares the type that the body returns. The manual's slip is
not an erratum of an expected output (the section has no example data), only a note.

Expected: 2 > 3 is false, so the result is `y`: `sc_r` = 3 (Integer).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
