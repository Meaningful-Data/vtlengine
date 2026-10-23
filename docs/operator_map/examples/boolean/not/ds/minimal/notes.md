One measure with the three values true, false and null.

The page lists `not false` as true, `not true` as false and `not null` as null.

The semantics sentence of the page, "returns true if op is false, otherwise true", has a typo
("otherwise true" would make `not true` true) and ignores null (`not null` is null): the listed rows
govern.

| Id_1 | Me_1 | result |
|---|---|---|
| 1 | true | false |
| 2 | false | true |
| 3 | null | null |
