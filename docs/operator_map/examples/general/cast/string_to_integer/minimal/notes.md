# `cast(Me_1, integer)` on a string measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to integer with an optional mask; this example gives none, so the default behaviour applies.

> To **integer**: First a `+` or `-` is possibly consumed to determine the sign of the integer; if it's not present, the sign is assumed to be positive. Finally, a sequence of digits is consumed to determine its absolute value; leading zeroes are skipped.

"When performing a conversion from a `string` value, all leading and trailing blanks are removed from the input string value before performing the conversion." The inputs have no blanks.

| Me_1 | Me_2 | why |
|---|---|---|
| `42` | `42` |  |
| `-7` | `-7` | the sign `-` is consumed |
| `+15` | `15` | the sign `+` is consumed |
| `007` | `7` | leading zeroes are skipped |
