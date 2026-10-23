# `cast(Me_1, number)` on a integer measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows integer to number implicitly.

> From **integer** to **number**: a `number` value is returned having the integer part equal to the `integer` value, and the decimal part equal to zero.

The conversion is implicit, and the example writes it as an explicit cast: "An **explicit cast without mask** can also be used to make an implicit cast more clear, and in this case its behavior remains the same."

| Me_1 | Me_2 | why |
|---|---|---|
| `5` | `5.0` | integer part 5, decimal part zero |
| `-3` | `-3.0` | integer part -3, decimal part zero |
