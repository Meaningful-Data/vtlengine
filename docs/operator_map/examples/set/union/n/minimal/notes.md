# union with three operands

Authored: the manual's two examples of `union` have two operands only. The syntax is
`union ( dsList )` with `dsList ::= ds { , ds }*`, and the manual says that the result
"contains the Data Points belonging to any of the operand Data Sets" and that, for Data Points
with the same Identifier values, "those Data Points are filtered by choosing the Data Point
belonging to the left most operand Data Set".

The measure of every input data point is `10 * Id_1 + k`, where `k` is 0 for `DS_1`, 1 for
`DS_2` and 2 for `DS_3`, so the last digit of a result value says which operand it came from.

Expected result, derived by hand for each identifier value:

| Id_1 | Found in | Leftmost operand | Me_1 |
|---|---|---|---|
| 1 | `DS_1` | `DS_1` | 10 |
| 2 | `DS_1`, `DS_2`, `DS_3` | `DS_1` | 20 |
| 3 | `DS_2`, `DS_3` | `DS_2` | 31 |
| 4 | `DS_3` | `DS_3` | 42 |
