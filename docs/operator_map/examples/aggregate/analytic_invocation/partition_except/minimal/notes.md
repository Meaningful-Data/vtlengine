# Analytic invocation with `partition except`

Authored: the manual has no example of `partition except` with identifiers. The syntax is `partition except identifier { , identifier }*`: the data set is partitioned by the identifiers that are not listed, here `Id_1`.

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

Sum of `Me_1` per partition of `Id_1`: `A` = 3 + 1 + 5 + 1 = 10, `B` = 4 + 8 = 12.
