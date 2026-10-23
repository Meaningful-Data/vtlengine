# `cast(Me_1, boolean)` on a number measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows number to boolean as an explicit cast without mask.

> From **number** to **boolean**: if the `number` absolute value is not equal to zero, then the boolean `true` literal is returned, in every other case the boolean `false` literal is returned.

| Me_1 | Me_2 | why |
|---|---|---|
| `0.0` | `false` | absolute value zero |
| `0.5` | `true` | absolute value not zero |
| `-3.25` | `true` | absolute value not zero |
