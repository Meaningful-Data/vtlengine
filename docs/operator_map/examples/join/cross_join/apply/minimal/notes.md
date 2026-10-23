# cross_join: apply

Authored. The `apply` clause adds the homonym measures `Me_1` of the two operands. Inputs: the cross pair, with the measure of `DS_2` named `Me_1` so that the two operands have a homonym measure: `DS_1` = (1, 10), (2, 20) on `Id_1`; `DS_2` = (A, 1), (B, 2) on `Id_2`.

"**apply** combines the homonym Measures in the source operands ... The expression *applyExpr* can use as input the names or aliases of the operand data sets. It applies the expression to each of the n-uple of homonym Measures in the input data sets producing in the target a single homonym Measure for each n-uple". Here that is `Me_1 := DS_1#Me_1 + DS_2#Me_1`, and "All the input measures do not appear". A sum with a null is null (User Manual: "Whenever a **null** value is involved in a mathematical operation (+, -, *, /, …), the result is **null**").

The cartesian product has 2 x 2 = 4 data points; `Me_1` is in both operands, so VDS₁ holds `DS_1#Me_1` and `DS_2#Me_1`.

| Id_1 | Id_2 | Me_1 |
|---|---|---|
| 1 | A | 11 |
| 1 | B | 12 |
| 2 | A | 21 |
| 2 | B | 22 |

Manual page: [Cross Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html).
