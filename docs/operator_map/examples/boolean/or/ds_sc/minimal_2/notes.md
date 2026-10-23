One measure with the three values true, false and null, and a boolean scalar.

`true` decides the result whatever the other operand is, null included: `null or true` is true.

The page lists `false or null` as null, `true or null` as true and `null or null` as null. The
operator is symmetric, so `null or false` is null and `null or true` is true.

| Id_1 | Me_1 | DS_r Me_1 |
|---|---|---|
| 1 | true | true |
| 2 | false | true |
| 3 | null | true |
