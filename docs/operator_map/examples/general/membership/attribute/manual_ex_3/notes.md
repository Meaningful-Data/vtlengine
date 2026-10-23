# `DS_1#At_1`: a non-viral attribute

Manual example 3 of [Membership](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html).
The values of `At_1` become a new measure named for its type, `string_var`, the measures of
`DS_1` are dropped, and `At_1` itself is dropped because it is not viral.

## Triage (both engines name the measure `str_var`)

Re-derived: the page says "A default conventional name is assigned to the new Measure depending
on its type: for example `num_var` if the Measure is `numeric`, `string_var` if it is `string`",
and the values agree. The only difference is the name. The typical behaviour's table of default
names adds: "These are only the names used in this manual for explanatory purposes and can be
personalised in the implementations. If VTL rules are exchanged, the personalised names need to be
shared with the partners of the exchange." vtlengine uses `str_var` (it lists it in
`docs/data_types.rst` for `cast`). The expected file keeps the manual's `string_var`, since no
recorded team decision adopts the personalised name; the comparison cannot accept two names.
