# `count` with `group by` and `having`, single-measure operand

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

`count` returns the number of Data Points ([Counting the number of data points](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Counting%20the%20number%20of%20data%20points/index.html)). The Aggregate invocation: "only groups for which havingCondition evaluates to true appear in the result".

The result has the single Integer measure `int_var` of the count page's result type, `dataset { measure<integer> int_var }`. `DS_1` has one measure and no null, so `int_var` is the number of data points of the group. The team decision [#1049](https://github.com/Meaningful-Data/vtlengine/issues/1049) gives the same output; the example carries no 📌 because its output does not go beyond the manual.

The condition `count ( Me_1 ) > 2` per group of `Id_1`: `A`: 4, `B`: 2. Only `A` passes, with `int_var` = 4. Numbers that are not whole are written with 6 decimals.
