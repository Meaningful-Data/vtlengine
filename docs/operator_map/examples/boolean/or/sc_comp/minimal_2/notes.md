One measure with the three values true, false and null, and a boolean scalar.

`true` decides the result whatever the other operand is, null included: `null or true` is true.

The page lists `false or null` as null, `true or null` as true and `null or null` as null.

The semantics sentence of the page, "returns true if at least one of the operands is true, otherwise
false", conflicts with its own null rows (`false or null` is null, not false): the listed rows
govern.

| Id_1 | Me_1 | Me_2 |
|---|---|---|
| 1 | true | true |
| 2 | false | true |
| 3 | null | true |
