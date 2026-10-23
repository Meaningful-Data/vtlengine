One measure with the three values true, false and null, and a boolean scalar.

`false` decides the result whatever the other operand is, null included: `null and false` is false.

The page lists `false and null` as false, `true and null` as null and `null and null` as null.

The semantics sentence of the page, "returns true if both operands are true, otherwise false",
conflicts with its own null rows (`true and null` is null, not false): the listed rows govern.

| Id_1 | Me_1 | Me_2 |
|---|---|---|
| 1 | true | false |
| 2 | false | false |
| 3 | null | false |
