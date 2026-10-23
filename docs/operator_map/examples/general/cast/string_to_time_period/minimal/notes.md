# `cast(Me_1, time_period)` on a string measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to time_period with an optional mask; this example gives none, so the default behaviour applies.

> To **time_period**: [...] exactly four digits are consumed to determine the year in the gregorian calendar. Then, a `-` is consumed if present. Then one of either `H`, `Q`, `M` or `D` is possibly consumed to determine the duration of the period; if neither of these are present, a `time_period` representing the entire determined year is returned. [...] If `Q` is found, a digit from 1 to 4 is consumed to determine which quarter is covered [...]. If `M` is found, `0` or `1` is possibly consumed, then another digit is consumed to determine which month is covered [...].

"When performing a conversion from a `string` value, all leading and trailing blanks are removed from the input string value before performing the conversion." The inputs have no blanks. The comparison takes `2001A` and `2001` as the same period.

| Me_1 | Me_2 | why |
|---|---|---|
| `2000Q1` | `2000Q1` | year, `Q`, quarter 1 |
| `2000-M03` | `2000M03` | year, `-`, `M`, `0`, month 3 |
| `2001` | `2001A` | year only: the entire year |
