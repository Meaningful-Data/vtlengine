One measure with the three values true, false and null, and a boolean scalar.

`false` leaves the other operand as it is, and null stays null (`null or false` is null).

The page lists `false or null` as null, `true or null` as true and `null or null` as null. A null on
the left of true or false is not listed: it follows from three-valued logic, which the User Manual
says VTL adopts (`v2.2/docs/user_manual/operators.rst`, section "Behaviour for Missing Data", item
"Boolean operations"). So `null or false` is null and `null or true` is true.

The semantics sentence of the page, "returns true if at least one of the operands is true, otherwise
false", conflicts with its own null rows (`false or null` is null, not false): the listed rows
govern.

| Id_1 | Me_1 | Me_2 |
|---|---|---|
| 1 | true | true |
| 2 | false | false |
| 3 | null | null |
