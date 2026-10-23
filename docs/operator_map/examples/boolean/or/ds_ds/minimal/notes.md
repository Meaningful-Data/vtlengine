Three-valued logic: each of the four data points that DS_1 and DS_2 have in common has a null
operand, on the left in rows 1 and 4 and on the right in rows 2 and 3. The two-valued cases are in
the manual example.

DS_2 also has a data point with Id_1 = 5 that DS_1 lacks. "Data Points that are not coupled are left
out (the inner join is used)" (Typical behaviours of the ML Operators, operators applicable on two
Scalar Values or Data Sets or Data Set Components), so the result has the four coupled data points
and none for Id_1 = 5.

The page lists `false or null` as null, `true or null` as true and `null or null` as null. A null on
the left of true or false is not listed: it follows from three-valued logic, which the User Manual
says VTL adopts (`v2.2/docs/user_manual/operators.rst`, section "Behaviour for Missing Data", item
"Boolean operations"). So `null or false` is null and `null or true` is true.

The semantics sentence of the page, "returns true if at least one of the operands is true, otherwise
false", conflicts with its own null rows (`false or null` is null, not false): the listed rows
govern.

| Id_1 | DS_1 Me_1 | DS_2 Me_1 | DS_r Me_1 (or) |
|---|---|---|---|
| 1 | null | false | null |
| 2 | true | null | true |
| 3 | false | null | null |
| 4 | null | true | true |
| 5 | (no data point) | true | (no data point) |
