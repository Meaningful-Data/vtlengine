Both operands are null. The page lists `false xor null`, `true xor null` and `null xor null` as
null.

The semantics sentence of the page, "returns true if only one of the operand is true (but not both),
false otherwise", conflicts with its own null rows (`false xor null` is null, not false): the listed
rows govern.

- `null xor null` is null
