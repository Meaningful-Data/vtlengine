# full_join: nary

Authored. The join of three operands.

Inputs (shared by the `inner_join`, `left_join` and `full_join` minimal examples): `DS_1` and
`DS_2` have the same identifiers `Id_1` (Integer) and `Id_2` (String) and one Number measure each,
`Me_1` and `Me_2`, so no component other than the identifiers is in both. The keys `(1,A)`,
`(2,A)` and `(2,B)` are in both; `(1,B)` is in `DS_1` only and `(1,C)` in `DS_2` only.

| Id_1 | Id_2 | DS_1 Me_1 | DS_2 Me_2 |
|---|---|---|---|
| 1 | A | 10 | 1 |
| 1 | B | 20 | |
| 1 | C | | 2 |
| 2 | A | 30 | 3 |
| 2 | B | 40 | 4 |

`DS_3` has the same identifiers and the measure `Me_3`: `(1,A)` 100, `(2,B)` 400, `(3,A)` 500.

All three operands have the same identifiers, so no `using` is needed. At each step the matched data points are joined, "Then, an additional data point is added to this set for each data point in the current join operand that wasn't matched before": the first step gives the five keys `(1,A)`, `(1,B)`, `(1,C)`, `(2,A)`, `(2,B)`, the second adds `(3,A)` from `DS_3`. Measures are null where their operand lacks the key.

| Id_1 | Id_2 | Me_1 | Me_2 | Me_3 |
|---|---|---|---|---|
| 1 | A | 10 | 1 | 100 |
| 1 | B | 20 | null | null |
| 1 | C | null | 2 | null |
| 2 | A | 30 | 3 | null |
| 2 | B | 40 | 4 | 400 |
| 3 | A | null | null | 500 |

Manual page: [Full Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html).
