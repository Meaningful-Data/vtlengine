# define operator: a parameter typed by a value domain

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `x` with the value domain `VD_1` as its type (Data Type
Syntax: `scalarType ::= { basicScalarType | valueDomainName | setName } ...`). `VD_1` is an
enumerated domain of strings (`value_domains/VD_1.json`: `A`, `B`, `C`).

[Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html): "VTL allows any value of a basic scalar type to be implicitly
converted into a value belonging to a value domain ... based on that basic scalar type
(downcast); obviously, the resulting value must be one of the allowed values", so the string
`"A"` is a value of `VD_1`; and in the body "an element of a value domain is implicitly upcast
into its corresponding basic scalar type", so `x || "_X"` is a string concatenation.

Expected: `SC_r` = `A_X` (String).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
