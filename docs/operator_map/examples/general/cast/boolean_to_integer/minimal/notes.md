# `cast(Me_1, integer)` on a boolean measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows boolean to integer as an explicit cast without mask.

> From **boolean** to **integer**: the integer `0` is returned if the `boolean` value is false, `1` if its true.

| Me_1 | Me_2 | why |
|---|---|---|
| `true` | `1` | true |
| `false` | `0` | false |
