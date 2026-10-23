# intersect with three operands

Authored: the manual's example of `intersect` has two operands only. The syntax is
`intersect ( dsList )` with `dsList ::= ds { , ds }*`, and the manual says that the result
"contains the Data Points belonging to all the operand Data Sets" and that, for Data Points
with the same Identifier values, "those Data Points are filtered by choosing the Data Point
belonging to the leftmost operand Data Set".

The measure of every input data point is `10 * Id_1 + k`, where `k` is 0 for `DS_1`, 1 for
`DS_2` and 2 for `DS_3`, so the last digit of a result value says which operand it came from.

Expected result, derived by hand for each identifier value:

| Id_1 | Found in | In all three? | Me_1 |
|---|---|---|---|
| 1 | `DS_1`, `DS_3` | no, `DS_2` lacks it | |
| 2 | `DS_1`, `DS_2` | no, `DS_3` lacks it | |
| 3 | `DS_1`, `DS_2`, `DS_3` | yes, the leftmost operand is `DS_1` | 30 |
| 4 | `DS_1`, `DS_2`, `DS_3` | yes, the leftmost operand is `DS_1` | 40 |
| 5 | `DS_2`, `DS_3` | no, `DS_1` lacks it | |

Ids 1, 2 and 5 are each in two of the three operands, and each lacks a different one (`DS_2`,
`DS_3` and `DS_1`). They stay out of the result, so leaving out any one operand would change it:
`intersect(DS_2, DS_3)` would add id 5, `intersect(DS_1, DS_3)` id 1 and `intersect(DS_1, DS_2)`
id 2. The example therefore shows that every operand takes part in the intersection.
