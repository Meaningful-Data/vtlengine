# `DS_2#Me_1`: a measure of a dataset with a viral attribute

Manual example 4 of [Membership](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html).
`Me_1` is kept, `Me_2` is dropped, and the viral attribute `At_1` is kept as viral ("viral
Attributes of `ds` are maintained in the result as viral, non-viral ones are dropped").

The manual's `ds_2.json` names its dataset `DS_1`, a copy-paste slip: the page says "Given the
operand datasets DS_1 and DS_2, where the attribute component At_1 is viral for DS_2", and only
`ds_2.json` has `At_1` as a `ViralAttribute`. The example loads that file as `DS_2`, the name the
script uses. This is an input naming slip, not an erratum of an expected output.
