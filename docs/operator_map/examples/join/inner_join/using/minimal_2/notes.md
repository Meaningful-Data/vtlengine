# inner_join: using (minimal_2)

Authored. The `using` clause names `Id_1` only, a subset of the identifiers that both operands have.

Inputs (own pair): `DS_1` and `DS_2` have the identifiers `Id_1` (Integer) and `Id_2` (String),
but no `(Id_1, Id_2)` key is in both, so the natural join on both identifiers would match nothing.
On `Id_1` alone, 1 is in both, 2 in `DS_1` only and 3 in `DS_2` only.

| DS_1 Id_1 | DS_1 Id_2 | DS_1 Me_1 |   | DS_2 Id_1 | DS_2 Id_2 | DS_2 Me_2 |
|---|---|---|---|---|---|---|
| 1 | A | 10 | | 1 | C | 1 |
| 1 | B | 20 | | 3 | D | 3 |
| 2 | B | 30 | | | | |

"if `using` is specified, **inner_join** removes from each of the I_i sets those identifiers not appearing in the `using` clause, and only uses the remaining identifiers as join keys if they appear in more than one dataset", so the only join key is `Id_1`. `Id_2` is in both operands but is not a key, so VDS₁ holds `d1#Id_2` and `d2#Id_2` ("The remaining components, identifier or not, coming from multiple data sets, which appear as many times as the data sets they come from"), still identifiers. Without the rename, the last step would strip the prefixes into two `Id_2` ("if this step determines a structure with homonymous components, an error is raised"), so the rename names them `Id_21` and `Id_22`.

`Id_1` = 1 is in both operands: each of the two data points of `DS_1` is paired with the one of `DS_2`. `Id_1` = 2 and 3 have no match. The natural join, on both identifiers, would give no data point.

| Id_1 | Id_21 | Id_22 | Me_1 | Me_2 |
|---|---|---|---|---|
| 1 | A | C | 10 | 1 |
| 1 | B | C | 20 | 1 |

Manual page: [Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html).
