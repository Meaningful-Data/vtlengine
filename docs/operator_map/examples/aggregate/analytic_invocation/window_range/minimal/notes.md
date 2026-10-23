# Analytic invocation with a `range` window

Authored: the manual has no example of `range`. The Analytic invocation says: "The keyword range means that the sliding windows includes all the Data Points whose values are in a certain range in respect to the value, for the current Data Point, of the Measure which the analytic is applied to", and num preceding / following give "the maximum difference to consider" (so the bounds are included). The analytic is applied to `Me_1` and ordered by `Me_1`, so the manual's rule and SQL's (the range is on the order key) give the same window: the data points of the partition whose `Me_1` is within 2 of the current one.

Input: the chapter's shared `DS_1` (`Id_1` String and `Id_2` Time_Period identifiers, `Me_1` and `Me_2` Number measures, no null):

| Id_1 | Id_2 | Me_1 | Me_2 |
|---|---|---|---|
| A | 2020Q1 | 3 | 1.5 |
| A | 2020Q2 | 1 | 3.5 |
| A | 2021Q1 | 5 | 2.5 |
| A | 2021Q2 | 1 | 0.5 |
| B | 2020Q1 | 4 | 6.5 |
| B | 2020Q2 | 8 | 4.5 |

Numbers that are not whole are written with 6 decimals.

Partition `A`, `Me_1` = 3, 1, 5, 1: for 3 the window is [1, 5] → 3 + 1 + 5 + 1 = 10; for 1 it is [-1, 3] → 3 + 1 + 1 = 5; for 5 it is [3, 7] → 3 + 5 = 8. Partition `B`, `Me_1` = 4, 8: for 4 the window is [2, 6] → 4; for 8 it is [6, 10] → 8. The values 1 and 5 sit exactly 2 away from 3, so this also checks that the bounds are included.
