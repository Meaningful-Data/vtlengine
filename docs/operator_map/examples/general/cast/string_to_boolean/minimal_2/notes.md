# `cast(Me_1, boolean)` on a string measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to boolean as an explicit cast without mask.

> From **string** to **boolean**: The `boolean` value representing true is returned if the input string, with any leading or trailing blanks removed, is equal to `"true"` not considering the case. The `boolean` value representing false is returned in any other case.

Neither string is `true`, so both give false. `1` is false too: the rule compares with the text `true` only. The strings have no blanks around them, because the loading of blanks is not what the example is about.

| Me_1 | Me_2 | why |
|---|---|---|
| `yes` | `false` | not `true`: any other case |
| `1` | `false` | not `true`: any other case |
