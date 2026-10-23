# `DS_2#At_1`: a viral attribute

Manual example 6 of [Membership](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html).
The values of `At_1` become the new measure `string_var` (its type is string), the measures of
`DS_2` are dropped, and `At_1` itself is kept as viral: "If `comp` is an Attribute, it follows
the Attribute propagation rule too".

The manual's `ds_2.json` names its dataset `DS_1`, a copy-paste slip: the page says "Given the
operand datasets DS_1 and DS_2, where the attribute component At_1 is viral for DS_2", and only
`ds_2.json` has `At_1` as a `ViralAttribute`. The example loads that file as `DS_2`, the name the
script uses. This is an input naming slip, not an erratum of an expected output.

## Triage (both engines return no measure)

Both engines return `Id_1`, `Id_2` and the viral `At_1` only, with no measure. Re-derived: the
page says "If `comp` is an Identifier or an Attribute Component in `ds`, then all the existing
Measures of `ds` are dropped in the result and a new Measure is added. The Data Points' values
for the new Measure are the same as the values of `comp` in `ds`", and the result has "a single
Measure". So the expected file stays.
