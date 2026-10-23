# `stddev_pop` with `group by` and `having`, single-measure operand

Authored: this example tests the `having` form on the simplest operand, a single measure. It is the script of `minimal` over the shared `DS_1` without its second measure `Me_2`. The manual applies the data set form to all the measures of the operand and does not restrict `having` to single-measure operands, so `minimal`, with two measures, is the general case.

Input `DS_1`:

| Id_1 | Id_2 | Me_1 |
|---|---|---|
| A | 2020Q1 | 3 |
| A | 2020Q2 | 1 |
| A | 2021Q1 | 5 |
| A | 2021Q2 | 1 |
| B | 2020Q1 | 4 |
| B | 2020Q2 | 8 |

`stddev_pop` returns the population standard deviation, sqrt(sum((x - mean)^2) / n) (the manual's ex_1 gives 1.632993 for 3, 5, 7) ([Population standard deviation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Population%20standard%20deviation/index.html)). The Aggregate invocation: "only groups for which havingCondition evaluates to true appear in the result".

The condition `stddev_pop ( Me_1 ) > 1.8` per group of `Id_1`: `A`: 1.658312, `B`: 2. Only `B` passes, with `Me_1` = 2. Numbers that are not whole are written with 6 decimals.
