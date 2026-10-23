# Analytic invocation with an unbounded window

Authored: the manual has no example of `unbounded`. The Analytic invocation says: "unbounded preceding means that the sliding window starts at the first Data Point of the partition" and "unbounded following indicates that the sliding window ends at the last Data Point of the partition". So the window is the whole partition, which differs from what an omitted window gives with an order clause ("range between unbounded preceding and current data point").

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

Sum of `Me_1` per partition: `A` = 3 + 1 + 5 + 1 = 10, `B` = 4 + 8 = 12, for every data point.
