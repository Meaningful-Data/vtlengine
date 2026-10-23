# `cast(Me_1, time)` on a string measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to time with an optional mask; this example gives none, so the default behaviour applies.

> To **time**: the input string is parsed, according to the ISO-8601 standard for time intervals, up to a precision of days, and the corresponding `time` value is returned.

"When performing a conversion from a `string` value, all leading and trailing blanks are removed from the input string value before performing the conversion." The inputs have no blanks. A `time` value is written `start/end`, as vtlengine reads it.

| Me_1 | Me_2 | why |
|---|---|---|
| `2020-01-01/2020-06-30` | `2020-01-01/2020-06-30` | ISO-8601 start/end |
| `2021-03-15/2021-03-20` | `2021-03-15/2021-03-20` | ISO-8601 start/end |
