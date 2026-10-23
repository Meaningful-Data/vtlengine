# `cast(Me_1, number)` on a boolean measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows boolean to number as an explicit cast without mask.

> From **boolean** to **number**: the number `0.0` is returned if the `boolean` value is false, `1.0` if its true.

| Me_1 | Me_2 | why |
|---|---|---|
| `true` | `1.0` | true |
| `false` | `0.0` | false |
