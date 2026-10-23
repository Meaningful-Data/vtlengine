# cross_join: calc_identifier

Authored. The `calc` clause adds the identifier `Id_3` with the constant value `"X"`. A constant is never null, so the decision of #1010 (an identifier calculated from a declared-nullable expression raises 1-1-1-16) does not apply.

Manual page: [Cross Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html).

Inputs (shared by the `cross_join` minimal examples): `DS_1` has the identifier `Id_1`
(Integer) and the measure `Me_1`; `DS_2` has the identifier `Id_2` (String) and the measure
`Me_2`. No component is in both operands, so the cartesian product needs no alias to be valid.

| DS_1 Id_1 | DS_1 Me_1 |   | DS_2 Id_2 | DS_2 Me_2 |
|---|---|---|---|---|
| 1 | 10 | | A | 1 |
| 2 | 20 | | B | 2 |

The operator performs "a *cartesian product* of the input operands", and "the final result has a size equal to the product of the sizes of each input data set": 2 x 2 = 4 data points, each component appearing once with its name and role.

The role of the calculated component "can be specified"; `Id_3` is a new component, so no identifier of the input is overwritten. Each data point of VDS₁ gets `Id_3` = X, and the keys stay unique.

| Id_1 | Id_2 | Id_3 | Me_1 | Me_2 |
|---|---|---|---|---|
| 1 | A | X | 10 | 1 |
| 1 | B | X | 10 | 2 |
| 2 | A | X | 20 | 1 |
| 2 | B | X | 20 | 2 |
