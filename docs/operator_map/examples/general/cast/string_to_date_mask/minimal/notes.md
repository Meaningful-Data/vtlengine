# `cast(Me_1, date, "YYYY-MM-DD")`: string to date with a mask

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to date with an optional mask; this example gives one.

> The [time formatting mask](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask): "The number of letters matches the number of digits, for example YYYY means that the year is represented with four digits and MM that the month is of 2 digits", and "Any other character can be used in the mask, meaning simply that it appears in the same position". Mask `YYYY-MM-DD`, the mask of the cast page's Example 2 (`cast(m1, date, "YYYY-MM-DD")`).

| Me_1 | Me_2 |
|---|---|
| `2020-01-15` | `2020-01-15` |
| `2021-12-31` | `2021-12-31` |
