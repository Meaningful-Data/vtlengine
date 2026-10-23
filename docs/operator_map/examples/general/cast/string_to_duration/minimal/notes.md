# `cast(Me_1, duration)` on a string measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to duration with an optional mask; this example gives none, so the default behaviour applies.

> To **duration**: If the input string starts with `P`, the entire string is consumed and parsed, according to the ISO-8601 standard for durations [...]. Otherwise, one of `A`, `S`, `Q`, `M`, `W` or `D` is consumed, and a `duration` value is returned representing the duration respectively of a year, a half of a year, a quarter of a year, a month, a week and a day.

"When performing a conversion from a `string` value, all leading and trailing blanks are removed from the input string value before performing the conversion." The inputs have no blanks. The expected durations are written with the same one-character codes.

| Me_1 | Me_2 | why |
|---|---|---|
| `A` | `A` | a year |
| `Q` | `Q` | a quarter |
| `D` | `D` | a day |
