# `DS_2#Id_1`: an identifier of a dataset with a viral attribute

Manual example 5 of [Membership](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html).
`Id_1` becomes the new measure `int_var` (its type is integer), the measures of `DS_2` are
dropped, and the viral attribute `At_1` is kept as viral.

The manual's `ds_2.json` names its dataset `DS_1`, a copy-paste slip: the page says "Given the
operand datasets DS_1 and DS_2, where the attribute component At_1 is viral for DS_2", and only
`ds_2.json` has `At_1` as a `ViralAttribute`. The example loads that file as `DS_2`, the name the
script uses. This is an input naming slip, not an erratum of an expected output.
