# `DS_1#At_1`: a non-viral attribute

Manual example 3 of [Membership](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html).
The values of `At_1` become a new measure named for its type, the measures of `DS_1` are
dropped, and `At_1` itself is dropped because it is not viral.

## Adaptation: the new measure is named `str_var`

The manual's `ex_3.json` names the new measure `string_var`:

```
{"name": "string_var", "role": "Measure", "data_type": "String"}
```

and its `ex_3.csv` has the header `Id_1,Id_2,string_var`.

The page says "A default conventional name is assigned to the new Measure depending on its type:
for example `num_var` if the Measure is `numeric`, `string_var` if it is `string`", and the
typical behaviour's table of default names adds: "These are only the names used in this manual for
explanatory purposes and can be personalised in the implementations. If VTL rules are exchanged,
the personalised names need to be shared with the partners of the exchange." vtlengine uses
`str_var`, and documents it in `docs/data_types.rst`. By the controller's ruling on default names,
the expected output names the measure `str_var`. The values are the manual's.
