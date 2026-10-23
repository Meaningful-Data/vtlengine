# cross_join: nary

Authored. The join of three operands.

Inputs (shared by the `cross_join` minimal examples): `DS_1` has the identifier `Id_1`
(Integer) and the measure `Me_1`; `DS_2` has the identifier `Id_2` (String) and the measure
`Me_2`. No component is in both operands, so the cartesian product needs no alias to be valid.

| DS_1 Id_1 | DS_1 Me_1 |   | DS_2 Id_2 | DS_2 Me_2 |
|---|---|---|---|---|
| 1 | 10 | | A | 1 |
| 2 | 20 | | B | 2 |

`DS_3` has the identifier `Id_3` (String) and the measure `Me_3`: (P, 100), (Q, 200).

"In each step, a data point in VDS₁ is generated for each pair of data points in the partial result and the joined data set. Then, the step is repeated by joining this partial result to the next data set. The final result has a size equal to the product of the sizes of each input data set": 2 x 2 x 2 = 8 data points.

| Id_1 | Id_2 | Id_3 | Me_1 | Me_2 | Me_3 |
|---|---|---|---|---|---|
| 1 | A | P | 10 | 1 | 100 |
| 1 | A | Q | 10 | 1 | 200 |
| 1 | B | P | 10 | 2 | 100 |
| 1 | B | Q | 10 | 2 | 200 |
| 2 | A | P | 20 | 1 | 100 |
| 2 | A | Q | 20 | 1 | 200 |
| 2 | B | P | 20 | 2 | 100 |
| 2 | B | Q | 20 | 2 | 200 |

Manual page: [Cross Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html).
