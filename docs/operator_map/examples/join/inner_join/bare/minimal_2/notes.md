# inner_join: bare (minimal_2)

Authored. The operands share the identifier `Id_2`, but neither has all the identifiers of the other: `DS_1` has `Id_1`, `Id_2` and `DS_2` has `Id_2`, `Id_3`.

| DS_1 Id_1 | DS_1 Id_2 | DS_1 Me_1 |   | DS_2 Id_2 | DS_2 Id_3 | DS_2 Me_2 |
|---|---|---|---|---|---|---|
| 1 | A | 10 | | A | X | 1 |
| 2 | B | 20 | | A | Y | 2 |
| | | | | C | X | 3 |

The page allows this: "**inner_join** requires ... that there exists a permutation of the join operands such that each operand has at least one identifier in common with any of the operands preceding it in that permutation", and "if `using` is not specified, **inner_join** performs a *natural join*, that is, it uses all the identifiers appearing in more than one dataset as join keys". The join key is `Id_2`. `Id_1` and `Id_3` each come from one operand: they "appear once and maintain their original name and role", so the result has three identifiers.

`Id_2` = A is in both: the data point of `DS_1` is paired with the two of `DS_2`. B and C have no match.

| Id_1 | Id_2 | Id_3 | Me_1 | Me_2 |
|---|---|---|---|---|
| 1 | A | X | 10 | 1 |
| 1 | A | Y | 10 | 2 |

Manual page: [Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html).
