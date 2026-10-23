# `median` with `group by` and `having`, single-measure operand

Authored, after the first run: the same script as `minimal`, over the shared `DS_1` without its second measure `Me_2`. The manual applies the data set form to all the measures of the operand and does not restrict `having` to single-measure operands, so `minimal` (two measures) stays the general case; this example shows the form on the simplest operand.

Input `DS_1`:

| Id_1 | Id_2 | Me_1 |
|---|---|---|
| A | 2020Q1 | 3 |
| A | 2020Q2 | 1 |
| A | 2021Q1 | 5 |
| A | 2021Q2 | 1 |
| B | 2020Q1 | 4 |
| B | 2020Q2 | 8 |

`median` returns the middle value of the sorted values, or the mean of the two middle values when there is an even number of them (the manual's ex_1 gives median(2, 4) = 3.0) ([Median value](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Median%20value/index.html)). The Aggregate invocation: "only groups for which havingCondition evaluates to true appear in the result".

The condition `median ( Me_1 ) > 3` per group of `Id_1`: `A`: 2, `B`: 6. Only `B` passes, with `Me_1` = 6. Numbers that are not whole are written with 6 decimals.
