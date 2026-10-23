Three-valued logic: every data point has a null operand, on the left in rows 1 and 4 and on the
right in rows 2 and 3.

The page lists `false or null` as null, `true or null` as true and `null or null` as null. A null on
the left of true or false is not listed: it follows from three-valued logic, which the User Manual
says VTL adopts (`v2.2/docs/user_manual/operators.rst`, section "Behaviour for Missing Data", item
"Boolean operations"). So `null or false` is null and `null or true` is true.

The semantics sentence of the page, "returns true if at least one of the operands is true, otherwise
false", conflicts with its own null rows (`false or null` is null, not false): the listed rows
govern.

| Id_1 | Me_1 | Me_2 | Me_3 = Me_1 or Me_2 |
|---|---|---|---|
| 1 | null | false | null |
| 2 | true | null | true |
| 3 | false | null | null |
| 4 | null | true | true |
