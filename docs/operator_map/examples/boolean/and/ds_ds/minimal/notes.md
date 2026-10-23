Three-valued logic: each of the four data points that DS_1 and DS_2 have in common has a null
operand, on the left in rows 1 and 4 and on the right in rows 2 and 3. The two-valued cases are in
the manual example.

DS_2 also has a data point with Id_1 = 5 that DS_1 lacks. "Data Points that are not coupled are left
out (the inner join is used)" (Typical behaviours of the ML Operators, operators applicable on two
Scalar Values or Data Sets or Data Set Components), so the result has the four coupled data points
and none for Id_1 = 5.

The page lists `false and null` as false, `true and null` as null and `null and null` as null. A
null on the left of true or false is not listed: it follows from three-valued logic, which the User
Manual says VTL adopts (`v2.2/docs/user_manual/operators.rst`, section "Behaviour for Missing Data",
item "Boolean operations"). So `null and false` is false and `null and true` is null.

The semantics sentence of the page, "returns true if both operands are true, otherwise false",
conflicts with its own null rows (`true and null` is null, not false): the listed rows govern.

| Id_1 | DS_1 Me_1 | DS_2 Me_1 | DS_r Me_1 (and) |
|---|---|---|---|
| 1 | null | false | false |
| 2 | true | null | null |
| 3 | false | null | false |
| 4 | null | true | null |
| 5 | (no data point) | true | (no data point) |
