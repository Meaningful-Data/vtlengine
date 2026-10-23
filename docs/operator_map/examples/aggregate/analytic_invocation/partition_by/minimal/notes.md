# Analytic invocation with `partition by`

Authored: the manual's examples have no `partition by`. The Analytic invocation says: "The input Data Set is partitioned according to the values of one or more Identifier Components", and with no window clause the window is the whole partition. Here the partition is by `Id_2`, the quarter.

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

Sum of `Me_1` per quarter: 2020Q1 = 3 + 4 = 7, 2020Q2 = 1 + 8 = 9, 2021Q1 = 5, 2021Q2 = 1.
