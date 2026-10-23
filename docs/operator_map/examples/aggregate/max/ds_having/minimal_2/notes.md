# `max` with `group by` and `having`, single-measure operand

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

`max` returns the greatest value; the result has the type of the operand (Number) ([Maximum value](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Maximum%20value/index.html)). The Aggregate invocation: "only groups for which havingCondition evaluates to true appear in the result".

The condition `max ( Me_1 ) < 6` per group of `Id_1`: `A`: 5, `B`: 8. Only `A` passes, with `Me_1` = 5. Numbers that are not whole are written with 6 decimals.
