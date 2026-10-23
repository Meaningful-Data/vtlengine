# full_join: using_nvl (minimal_2)

Authored. The `using` clause names `Id_1` only, and two `nvl()` rules name the `Id_2` of each operand through its alias, with different values.

Inputs (own pair): `DS_1` and `DS_2` have the identifiers `Id_1` (Integer) and `Id_2` (String),
but no `(Id_1, Id_2)` key is in both, so the natural join on both identifiers would match nothing.
On `Id_1` alone, 1 is in both, 2 in `DS_1` only and 3 in `DS_2` only.

| DS_1 Id_1 | DS_1 Id_2 | DS_1 Me_1 |   | DS_2 Id_1 | DS_2 Id_2 | DS_2 Me_2 |
|---|---|---|---|---|---|---|
| 1 | A | 10 | | 1 | C | 1 |
| 1 | B | 20 | | 3 | D | 3 |
| 2 | B | 30 | | | | |

The `using` clause "must specify a subset of the identifiers common to every join operand. For every other identifier not included in the using clause, a nvl() expression must be specified": `Id_2` may be missing on either side, so each operand's `Id_2` gets a rule. VDS₁ has the key `Id_1` once, `Me_1` and `Me_2` (each from one operand) and, since `Id_2` is in both operands but not a key, `d1#Id_2` and `d2#Id_2`. The rename names them `Id_21` and `Id_22`, as the last step would otherwise give two `Id_2`. "Note that the same nvl() rule is applied to the same identifier when appearing in multiple join operands; prefixing the identifier name with the alias of the operand in the nvl() rule allows to specify different rules for each of them."

The full outer join pairs the data points with `Id_1` = 1. `Id_1` = 2 is in `DS_1` only: "the missing Identifiers assume the value specified in the relevant **nvl()** clause, while the Measures take the null value", so `Id_22` = X (the rule of `d2#Id_2`) and `Me_2` = null. `Id_1` = 3 is in `DS_2` only, and "an additional data point is added" for it, with `Id_21` = Y (the rule of `d1#Id_2`) and `Me_1` = null.

| Id_1 | Id_21 | Id_22 | Me_1 | Me_2 |
|---|---|---|---|---|
| 1 | A | C | 10 | 1 |
| 1 | B | C | 20 | 1 |
| 2 | B | X | 30 | null |
| 3 | Y | D | null | 3 |

Manual page: [Full Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html).
