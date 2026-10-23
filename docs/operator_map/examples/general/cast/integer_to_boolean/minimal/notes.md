# `cast(Me_1, boolean)` on a integer measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows integer to boolean as an explicit cast without mask.

> From **integer** to **boolean**: if the `integer` value is not equal to zero, then the boolean `true` literal is returned, in every other case the boolean `false` literal is returned.

| Me_1 | Me_2 | why |
|---|---|---|
| `0` | `false` | equal to zero |
| `7` | `true` | not zero |
| `-2` | `true` | not zero |
