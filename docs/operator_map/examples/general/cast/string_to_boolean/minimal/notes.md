# `cast(Me_1, boolean)` on a string measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to boolean as an explicit cast without mask.

> From **string** to **boolean**: The `boolean` value representing true is returned if the input string, with any leading or trailing blanks removed, is equal to `"true"` not considering the case. The `boolean` value representing false is returned in any other case.

The second example, `minimal_2`, holds strings that are neither `true` nor `false`.

| Me_1 | Me_2 | why |
|---|---|---|
| `true` | `true` | equal to `true` |
| `TRUE` | `true` | equal to `true` not considering the case |
| `false` | `false` | any other case |
