# inner_join: filter

Authored. The `filter` clause keeps the joined data points with `Me_1 > 15`.

Manual page: [Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html).

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

The join keys are the identifiers in both operands, `Id_1` and `Id_2` (natural join: "if `using` is not specified, **inner_join** performs a *natural join*, that is, it uses all the identifiers appearing in more than one dataset as join keys"). The data points are matched "according to SQL inner join", so VDS₁ holds the keys found in both operands: `(1,A)`, `(2,A)`, `(2,B)`.

The filter "is applied on VDS₁ ... when the expression is **true** the Data Point is kept in the result, otherwise it is discarded". A comparison with a null is null (User Manual: "Whenever a **null** value is involved in a comparison ... the result of the comparison is **null**"), and in a filter "the Data Points for which *p* is **false** or **null** are discarded".

| Id_1 | Id_2 | Me_1 | Me_2 |
|---|---|---|---|
| 2 | A | 30 | 3 |
| 2 | B | 40 | 4 |
