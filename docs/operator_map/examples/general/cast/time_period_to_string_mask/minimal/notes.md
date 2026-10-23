# `cast(Me_1, string, "YYYY\MMM")`: time_period to string with a mask

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows time_period to string with an optional mask; this example gives one.

> The [time formatting mask](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask): "The number of letters matches the number of digits, for example YYYY means that the year is represented with four digits and MM that the month is of 2 digits", and "Any other character can be used in the mask, meaning simply that it appears in the same position". Mask `YYYY\MMM`, given for a "Monthly period: "2015M03"". `\M` is the letter `M`, because special characters "prefixed by the reverse slash [...] appear in the same position", and `MM` is the month in two digits.

| Me_1 | Me_2 | why |
|---|---|---|
| `2015M03` | `2015M03` | March 2015 |
| `2020M11` | `2020M11` | November 2020 |
