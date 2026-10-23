The operand is null. The page lists `not false` as true, `not true` as false and `not null` as null.

The semantics sentence of the page, "returns true if op is false, otherwise true", has a typo
("otherwise true" would make `not true` true) and ignores null (`not null` is null): the listed rows
govern.

- `not null` is null
