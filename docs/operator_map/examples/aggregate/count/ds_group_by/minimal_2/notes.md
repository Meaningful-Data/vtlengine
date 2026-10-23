# `count` with `group by` over data points with null measures

Authored: this example tests which data points `count` counts when their measures are null.
The input is the chapter's shared `DS_1` with nulls put in two data points of `A`:

| Id_1 | Id_2 | Me_1 | Me_2 |
|---|---|---|---|
| A | 2020Q1 | 3 | 1.5 |
| A | 2020Q2 | null | null |
| A | 2021Q1 | 5 | null |
| A | 2021Q2 | 1 | 0.5 |
| B | 2020Q1 | 4 | 6.5 |
| B | 2020Q2 | 8 | 4.5 |

The manual does not settle which of these data points `count` counts:

- the count page ([Counting the number of data points](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Counting%20the%20number%20of%20data%20points/index.html))
  says "The operator returns the number of the input Data Points", which read literally counts
  all four data points of `A`;
- the User Manual says "the input Data Points having a null value are in general not
  considered", which read literally drops both 2020Q2 and 2021Q1 and counts two.

The expected output follows the text of the team decision
[#1049](https://github.com/Meaningful-Data/vtlengine/issues/1049) (📌): "A Data Point whose
Measures are all null is ignored; a Data Point holding at least one non-null Measure is counted",
and the data set form "always returns a single Measure `int_var` of type Integer, whatever the
number of Measures of the operand". It goes beyond the manual on the second point of the
criterion, so the example carries `decision=1049`.

| Id_1 | Data point | Measures | Counted |
|---|---|---|---|
| A | 2020Q1 | 3, 1.5 | yes |
| A | 2020Q2 | null, null | no: all its measures are null |
| A | 2021Q1 | 5, null | yes: one measure is not null |
| A | 2021Q2 | 1, 0.5 | yes |
| B | 2020Q1 | 4, 6.5 | yes |
| B | 2020Q2 | 8, 4.5 | yes |

So `int_var` is 3 for `A` (the two literal readings above give 4 and 2) and 2 for `B`.
