# filter with a null condition

The manual: the operator "filters out the Data Points of the operand Data Set for which
*filterCondition* condition evaluates to **false** or **null**".

| Id_1 | Me_1 | Me_1 > 10 | kept |
|---|---|---|---|
| 1 | 5 | false | no |
| 2 | null | null | no |
| 3 | 15 | true | yes |
| 4 | 20 | true | yes |
