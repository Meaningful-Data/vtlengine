# `avg` with `group by` over a null value

Authored (review fix round 1): the null axis of `avg`. The input is the shared `DS_1` reduced
to its first measure, with the `Me_1` of `A, 2021Q1` null:

| Id_1 | Id_2 | Me_1 |
|---|---|---|
| A | 2020Q1 | 3 |
| A | 2020Q2 | 1 |
| A | 2021Q1 | null |
| A | 2021Q2 | 1 |
| B | 2020Q1 | 4 |
| B | 2020Q2 | 8 |

The User Manual says: "the input Data Points having a null value are in general not
considered. In the average, for example, they are not considered both in the numerator (the sum)
and in the denominator (the count)." The page ([Average value](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Average%20value/index.html))
says "The result is always a number".

- `A`: the non-null values are 3, 1 and 1, so the average is (3 + 1 + 1) / 3 = 5/3 = 1.666667,
  written with 6 decimals. Counting the null in the denominator would give 5/4 = 1.25.
- `B`: (4 + 8) / 2 = 6.
