Both operands are null. The page lists `false and null` as false, `true and null` as null and
`null and null` as null.

The semantics sentence of the page, "returns true if both operands are true, otherwise false",
conflicts with its own null rows (`true and null` is null, not false): the listed rows govern.

- `null and null` is null
