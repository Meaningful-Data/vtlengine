# Analytic invocation with a window ending at the current data point

Authored: the manual has no example of `current data point`. The Analytic invocation says: "current data point specifies that the window starts or ends at the current Data Point", and with data points, 1 preceding is the data point before the current one in the order of the order clause.

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

Partition `A` in the order of `Id_2`, `Me_1` = 3, 1, 5, 1: 3, 3 + 1 = 4, 1 + 5 = 6, 5 + 1 = 6. Partition `B`, `Me_1` = 4, 8: 4, 4 + 8 = 12.
