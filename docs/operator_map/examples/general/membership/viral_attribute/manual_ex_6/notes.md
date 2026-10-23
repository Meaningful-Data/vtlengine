# `DS_2#At_1`: a viral attribute

Manual example 6 of [Membership](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html).
The values of `At_1` become a new measure named for its type, the measures of `DS_2` are
dropped, and `At_1` itself is kept as viral: "If `comp` is an Attribute, it follows the
Attribute propagation rule too".

## Adaptation: the new measure is named `str_var`

The manual's `ex_6.json` names the new measure `string_var`:

```
{"name": "string_var", "role": "Measure", "data_type": "String"}
```

and its `ex_6.csv` has the header `Id_1,Id_2,string_var,At_1`.

The page says "A default conventional name is assigned to the new Measure depending on its type:
for example `num_var` if the Measure is `numeric`, `string_var` if it is `string`", and the
typical behaviour's table of default names adds: "These are only the names used in this manual for
explanatory purposes and can be personalised in the implementations. If VTL rules are exchanged,
the personalised names need to be shared with the partners of the exchange." The expected output
names the measure `str_var`, vtlengine's default name for a string measure, which it documents in
`docs/data_types.rst`. The values are the manual's.

## Input

The manual's `ds_2.json` names its dataset `DS_1`, a copy-paste slip: the page says "Given the
operand datasets DS_1 and DS_2, where the attribute component At_1 is viral for DS_2", and only
`ds_2.json` has `At_1` as a `ViralAttribute`. The example loads that file as `DS_2`, the name the
script uses. This is an input naming slip, not an erratum of an expected output.

## The result keeps a measure

The viral `At_1` is kept, and the result still has a measure of its own: the page says "If
`comp` is an Identifier or an Attribute Component in `ds`, then all the existing Measures of `ds`
are dropped in the result and a new Measure is added. The Data Points' values for the new Measure
are the same as the values of `comp` in `ds`", and the result has "a single Measure". A result
with `Id_1`, `Id_2` and `At_1` only would have no measure.
