# define operator: a parameter of type `set<string>`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `s` with the set type `set<string>` (Data Type Syntax:
`scalarSetType ::= set { < scalarType > }`). The call passes the set literal `{"A", "B"}`,
a value of that type: in [Element of](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Element%20of/index.html), a set "can be a Set name or a Set literal", and
its type is `set<scalar>`. The body tests membership with `in`.

Expected: `sc_r` = true (Boolean), because `"B"` is an element of the set `{"A", "B"}`.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
