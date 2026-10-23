# cross_join: calc_viral_attribute

Authored. The `calc` clause adds the viral attribute `At_1` with the constant value `"EP"`.

Manual page: [Cross Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html).

Inputs (shared by the `cross_join` minimal examples): `DS_1` has the identifier `Id_1`
(Integer) and the measure `Me_1`; `DS_2` has the identifier `Id_2` (String) and the measure
`Me_2`. No component is in both operands, so the cartesian product needs no alias to be valid.

| DS_1 Id_1 | DS_1 Me_1 |   | DS_2 Id_2 | DS_2 Me_2 |
|---|---|---|---|---|
| 1 | 10 | | A | 1 |
| 2 | 20 | | B | 2 |

The operator performs "a *cartesian product* of the input operands", and "the final result has a size equal to the product of the sizes of each input data set": 2 x 2 = 4 data points, each component appearing once with its name and role.

Each data point of VDS₁ gets `At_1` = EP, with the role Viral Attribute. On the viral attributes the page says: "The Attributes explicitly calculated through the **calc** or **aggr** clauses are maintained unchanged."

| Id_1 | Id_2 | Me_1 | Me_2 | At_1 |
|---|---|---|---|---|
| 1 | A | 10 | 1 | EP |
| 1 | B | 10 | 2 | EP |
| 2 | A | 20 | 1 | EP |
| 2 | B | 20 | 2 | EP |
