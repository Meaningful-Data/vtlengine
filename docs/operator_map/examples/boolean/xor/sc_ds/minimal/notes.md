One measure with the three values true, false and null, and a boolean scalar.

`xor true` flips the value of the operand and keeps null: a null operand gives null whatever the
other operand is.

The page lists `false xor null`, `true xor null` and `null xor null` as null.

The semantics sentence of the page, "returns true if only one of the operand is true (but not both),
false otherwise", conflicts with its own null rows (`false xor null` is null, not false): the listed
rows govern.

| Id_1 | Me_1 | DS_r Me_1 |
|---|---|---|
| 1 | true | false |
| 2 | false | true |
| 3 | null | null |
