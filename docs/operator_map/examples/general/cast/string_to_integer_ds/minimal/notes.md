# `cast(DS_1, integer)` on a dataset with a string measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), whose operand may be `dataset{ measure<scalar> _ }`.
The typical behaviour of the [operators changing the data
type](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#operators-changing-the-data-type) applies: "the operators
changing the data type are allowed only on mono-measure operand Data Sets [...]. A default generic
Measure is assigned by default to the result Data Set, depending on the data type of the result",
and the default measure of an integer is `int_var`.

The values follow the string to integer rule: "First a `+` or `-` is possibly consumed to
determine the sign of the integer [...] Finally, a sequence of digits is consumed to determine its
absolute value".

| Id_1 | Me_1 | int_var |
|---|---|---|
| 1 | `42` | 42 |
| 2 | `-7` | -7 |
