# `cast(Me_1, VD_1)`: from one value domain to another

Authored after the cast page's Example 8, "From GEO_AREA2 to GEO_AREA1": `ds2 := ds1 [ calc m2 :=
cast (GEO, GEO_AREA1) ];`, where "Component GEO is of type GEO_AREA2, then we specify it has to be
cast into GEO_AREA1. As both work on *strings* (and the values are compatible), the conversion is
feasible."

The [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html) page lets `scalarType` be any "scalar type (see the section:
Data type syntax)", and that syntax is
`scalarType ::= { basicScalarType | valueDomainName | setName } ...`, so a value domain can be
the target of a cast.

The rule: "Since an element of a value domain is implicitly upcast into its corresponding basic
scalar type, and in turn that value can be implictly downcast into another value domain based on
that scalar type (or one of its subsets), an implicit conversion between those two domains is
straightforward. Of course, if the the resulting value is not one of the allowed values of that
particular domain or subset, an error is raised."

`Me_1` takes its values from `VD_2` (`A`, `B`): its structure says so with the `subset` property
of vtlengine's structure format, on the basic type `String`. `VD_1` has the values `A`, `B` and
`C`. Each value is upcast to its string and downcast to `VD_1`, where it belongs, so it is kept as
it is. The expected type of `Me_2` is `String`, the basic type of `VD_1`.

| Id_1 | Me_1 (VD_2) | Me_2 (VD_1) |
|---|---|---|
| 1 | `A` | `A` |
| 2 | `B` | `B` |
