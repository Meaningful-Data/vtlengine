# `cast(Me_1, string, "YYYY-MM-DD/YYYY-MM-DD")`: time to string with a mask

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows time to string with an optional mask; this example gives one.

> The [time formatting mask](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask): "The number of letters matches the number of digits, for example YYYY means that the year is represented with four digits and MM that the month is of 2 digits", and "Any other character can be used in the mask, meaning simply that it appears in the same position". Mask `YYYY-MM-DD/YYYY-MM-DD`, given for time values with "Start and end, such as "20150303/20180405"".

The manual's sample value has no `-`, which its mask does: by the rule above, the mask writes `2015-03-03/2018-04-05`. The example follows the mask.

| Me_1 | Me_2 | why |
|---|---|---|
| `2015-03-03/2018-04-05` | `2015-03-03/2018-04-05` | start and end |
| `2020-01-01/2020-06-30` | `2020-01-01/2020-06-30` | start and end |
