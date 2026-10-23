# `cast(Me_1, string)` on a number measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows number to string with an optional mask; this example gives none, so the default behaviour applies.

> From **number**: the `number` value may be printed either in the scientific notation, or as a sequence of digits, possibly preceded by a `-`, with `.` as the decimal separator, without any leading zeroes or thousands separators.

The expected strings are the sequence-of-digits form. The manual also allows the scientific notation, which a string comparison cannot accept as well, so a scientific result would be a mismatch that the manual allows. The values are exact in binary floating point and are not between -1 and 1, so the reading of "leading zeroes" does not matter.

| Me_1 | Me_2 | why |
|---|---|---|
| `3.5` | `3.5` |  |
| `-12.75` | `-12.75` | preceded by - |
