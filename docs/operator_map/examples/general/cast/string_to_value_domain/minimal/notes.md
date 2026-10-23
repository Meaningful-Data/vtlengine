# `cast(Me_1, VD_1)`: from a string to a value domain

Authored after the cast page's Example 6, "From string to GEO_AREA": `ds2 := ds1[calc m2 :=
cast(GEO_STRING, GEO_AREA)];`, where "GEO_STRING be a string Component of Data Set *ds1* with
string values compatible with the GEO_AREA Value Domain Subset".

The [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html) page lets `scalarType` be any "scalar type (see the section:
Data type syntax)", and that syntax is
`scalarType ::= { basicScalarType | valueDomainName | setName } ...`, so a value domain can be
the target of a cast.

The rule is the downcast: "VTL allows any value of a basic scalar type to be implicitly converted
into a value belonging to a value domain or value domain subset based on that basic scalar type
(downcast); obviously, the resulting value must be one of the allowed values of that particular
domain or subset; otherwise, an error is raised."

`VD_1` (in `value_domains/`) is a string value domain with the values `A`, `B` and `C`. Every input
value belongs to it, so each one is kept as it is. A component of a string value domain has the
basic type `String` in vtlengine's structures, and that is the expected type of `Me_2`.

| Id_1 | Me_1 | Me_2 |
|---|---|---|
| 1 | `A` | `A`, in VD_1 |
| 2 | `B` | `B`, in VD_1 |
