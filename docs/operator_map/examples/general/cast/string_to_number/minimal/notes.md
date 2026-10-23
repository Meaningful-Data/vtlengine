# `cast(Me_1, number)` on a string measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to number with an optional mask; this example gives none, so the default behaviour applies.

> To **number**: First a `+` or `-` is possibly consumed to determine the sign of the number; [...] Then a sequence of digits is consumed to determine the absolute value of the integer part of the number; leading zeroes are skipped. Then a `.` is possibly consumed; if it's not present, the parsing ends and the number is returned without a decimal part. Otherwise, a sequence of digits is consumed to determine its decimal part. Then `E` is possibly consumed to determine if a base-ten exponent is present; [...] Finally a sequence of digits is consumed to determine the exponent, then the number is returned multiplied by the corresponding power of 10.

"When performing a conversion from a `string` value, all leading and trailing blanks are removed from the input string value before performing the conversion." The inputs have no blanks. By that rule an exponent needs a `.` before it, which `1.5E2` has.

| Me_1 | Me_2 | why |
|---|---|---|
| `3.25` | `3.25` | integer part, `.`, decimal part |
| `-0.5` | `-0.5` | sign, integer part 0, decimal part |
| `1.5E2` | `150` | 1.5 multiplied by 10 to the power 2 |
| `+2` | `2` | no `.`: the parsing ends, no decimal part |
