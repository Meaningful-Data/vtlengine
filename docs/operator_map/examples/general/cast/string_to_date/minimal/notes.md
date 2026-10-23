# `cast(Me_1, date)` on a string measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to date with an optional mask; this example gives none, so the default behaviour applies.

> To **date**: First, exactly four digits are consumed to determine the year in the gregorian calendar. Then a `-` is consumed. Then, exactly two digits are consumed to determine the month in the range 01-12. Then a `-` is consumed again. Finally, exactly two digits are consumed to determine the day of the month [...].

"When performing a conversion from a `string` value, all leading and trailing blanks are removed from the input string value before performing the conversion." The inputs have no blanks.

| Me_1 | Me_2 |
|---|---|
| `2020-01-15` | `2020-01-15` |
| `2021-12-31` | `2021-12-31` |
