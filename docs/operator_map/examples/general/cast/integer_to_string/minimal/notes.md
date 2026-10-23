# `cast(Me_1, string)` on an integer measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows integer to string with an optional mask; this example gives none, so the default behaviour applies.

> From **integer**: the `integer` value is printed as a sequence of digits, without any leading zeroes or thousands separators, possibly preceded by a `-`.

| Me_1 | Me_2 | why |
|---|---|---|
| `1500` | `1500` | no thousands separator |
| `-7` | `-7` | preceded by - |
| `0` | `0` |  |
