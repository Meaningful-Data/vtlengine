# Analytic invocation with `order by ... desc`

Authored: the manual's examples do not write `asc` or `desc`. The Analytic invocation says: "The input Data Set is ordered according to the values of one or more Components, in ascending order if asc is specified, in descending order if desc is specified". The rank page: "If some Data Points are in the same order according to the specified orderClause, the same order number (rank) is assigned and a gap appears in the sequence of the assigned ranks".

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

Partition `A`, `Me_1` = 3 (2020Q1), 1 (2020Q2), 5 (2021Q1), 1 (2021Q2), in descending order: 5 → rank 1, 3 → rank 2, 1, 1 → rank 3 (tie). Partition `B`, `Me_1` = 4 (2020Q1), 8 (2020Q2): 8 → 1, 4 → 2.
