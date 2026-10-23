One measure with the three values true, false and null, and a boolean scalar.

`xor true` flips the value of the operand and keeps null: a null operand gives null whatever the
other operand is.

The page lists `false xor null`, `true xor null` and `null xor null` as null. The operator is
symmetric, so a null operand on the left gives null as well.

| Id_1 | Me_1 | DS_r Me_1 |
|---|---|---|
| 1 | true | false |
| 2 | false | true |
| 3 | null | null |
