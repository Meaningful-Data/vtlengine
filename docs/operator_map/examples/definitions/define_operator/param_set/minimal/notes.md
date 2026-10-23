# define operator: a parameter of type `set<string>`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `s` with the set type `set<string>` (Data Type Syntax:
`scalarSetType ::= set { < scalarType > }`). The call passes the set by its name: the
enumerated value domain `VD_AB` (`value_domains/VD_AB.json`, the strings `A` and `B`). In
[Element of](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Element%20of/index.html) a set "can be a Set name or a Set literal" and a collection is "the Set
or the Value Domain which contains the values"; a set literal cannot be the argument of a call,
since the VTL grammar takes set literals only after `in` and `not_in`. The body tests
membership with `in`.

Expected: `SC_r` = true (Boolean), because `"B"` is one of the values of `VD_AB`.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
