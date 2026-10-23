# `var_pop` with `group by` and `having`, single-measure operand

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

`var_pop` returns the population variance, sum((x - mean)^2) / n (the manual's ex_1 gives 2.666667 for 3, 5, 7) ([Population variance](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Population%20variance/index.html)). The Aggregate invocation: "only groups for which havingCondition evaluates to true appear in the result".

The condition `var_pop ( Me_1 ) < 3` per group of `Id_1`: `A`: 2.750000, `B`: 4. Only `A` passes, with `Me_1` = 2.750000. Numbers that are not whole are written with 6 decimals.
