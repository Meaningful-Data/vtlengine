# `cast(Me_1, time_period, "YYYY\QQ")`: string to time_period with a mask

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to time_period with an optional mask; this example gives one.

> The [time formatting mask](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask): "The number of letters matches the number of digits, for example YYYY means that the year is represented with four digits and MM that the month is of 2 digits", and "Any other character can be used in the mask, meaning simply that it appears in the same position". Mask `YYYY\QQ`, from the cast page's Example 9: `cast ( "2000Q1", time_period, "YYYY\QQ" )`, "The first quarter of year 2000". `\Q` is the letter `Q` and `Q` the quarter.

| Me_1 | Me_2 | why |
|---|---|---|
| `2000Q1` | `2000Q1` | the first quarter of 2000 |
| `2001Q4` | `2001Q4` |  |
