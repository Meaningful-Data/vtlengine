One operand is null, on the left or on the right. The page lists `false and null` as false,
`true and null` as null and `null and null` as null. A null on the left of true or false is not
listed: it follows from three-valued logic, which the User Manual says VTL adopts
(`v2.2/docs/user_manual/operators.rst`, section "Behaviour for Missing Data", item "Boolean
operations"). So `null and false` is false and `null and true` is null.

The semantics sentence of the page, "returns true if both operands are true, otherwise false",
conflicts with its own null rows (`true and null` is null, not false): the listed rows govern.

- `true and null` is null
- `null and true` is null
- `false and null` is false
- `null and false` is false
