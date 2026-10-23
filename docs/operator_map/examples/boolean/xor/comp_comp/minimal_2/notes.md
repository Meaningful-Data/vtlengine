Three-valued logic: every data point has a null operand, on the left in rows 1 and 4 and on the
right in rows 2 and 3.

The page lists `false xor null`, `true xor null` and `null xor null` as null. A null on the left of
true or false is not listed: it follows from three-valued logic, which the User Manual says VTL
adopts (`v2.2/docs/user_manual/operators.rst`, section "Behaviour for Missing Data", item "Boolean
operations"). So `null xor false` and `null xor true` are null.

The semantics sentence of the page, "returns true if only one of the operand is true (but not both),
false otherwise", conflicts with its own null rows (`false xor null` is null, not false): the listed
rows govern.

| Id_1 | Me_1 | Me_2 | Me_3 = Me_1 xor Me_2 |
|---|---|---|---|
| 1 | null | false | null |
| 2 | true | null | null |
| 3 | false | null | null |
| 4 | null | true | null |
