# `cast(Me_1, string, "YYYYMMDD")`: date to string with a mask

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows date to string with an optional mask; this example gives one.

> The [time formatting mask](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask): "The number of letters matches the number of digits, for example YYYY means that the year is represented with four digits and MM that the month is of 2 digits", and "Any other character can be used in the mask, meaning simply that it appears in the same position". Mask `YYYYMMDD`, from the same section: "YYYMMDD means four digits for the year followed by two digits for the month and then two digits for the day (e.g., 20180405 means the year 2018, month April, day 5th)".

The manual's `YYYMMDD` has three `Y` for four digits; the mask has four, one per digit.

| Me_1 | Me_2 |
|---|---|
| `2018-04-05` | `20180405` |
| `2020-01-15` | `20200115` |
