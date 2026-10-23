# `cast(Me_1, integer)` on a string measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to integer with an optional mask; this example gives none, so the default behaviour applies.

> To **integer**: First a `+` or `-` is possibly consumed to determine the sign of the integer; if it's not present, the sign is assumed to be positive. Finally, a sequence of digits is consumed to determine its absolute value; leading zeroes are skipped.

"When performing a conversion from a `string` value, all leading and trailing blanks are removed from the input string value before performing the conversion." The inputs have no blanks.

For a null: "An empty string generally causes an error to be raised when instead a `null`
input string is converted to the corresponding `null` value of the desired value domain." The
last data point has a null `Me_1` (an empty field of the CSV), so its `Me_2` is null.

| Me_1 | Me_2 | why |
|---|---|---|
| `42` | `42` |  |
| `-7` | `-7` | the sign `-` is consumed |
| `+007` | `7` | the sign `+` is consumed, and leading zeroes are skipped |
| null | null | a null string gives a null integer |
