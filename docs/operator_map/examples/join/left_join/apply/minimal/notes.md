# left_join: apply

Authored. The `apply` clause adds the homonym measures `Me_1` of the two operands. Inputs: the shared pair, with the measure of `DS_2` named `Me_1` so that the two operands have a homonym measure (same keys and values: `DS_2` `Me_1` = 1, 2, 3, 4 at `(1,A)`, `(1,C)`, `(2,A)`, `(2,B)`).

"**apply** combines the homonym Measures in the source operands ... The expression *applyExpr* can use as input the names or aliases of the operand data sets. It applies the expression to each of the n-uple of homonym Measures in the input data sets producing in the target a single homonym Measure for each n-uple". Here that is `Me_1 := DS_1#Me_1 + DS_2#Me_1`, and "All the input measures do not appear". A sum with a null is null (User Manual: "Whenever a **null** value is involved in a mathematical operation (+, -, *, /, …), the result is **null**").

The join keys are `Id_1`, `Id_2`; the left join keeps the four keys of `DS_1`, with `DS_2#Me_1` null at `(1,B)`.

| Id_1 | Id_2 | Me_1 |
|---|---|---|
| 1 | A | 11 |
| 1 | B | null |
| 2 | A | 33 |
| 2 | B | 44 |

Manual page: [Left Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html).
