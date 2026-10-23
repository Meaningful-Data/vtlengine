# Analytic invocation with `partition except all`

Authored: this example tests `partition except all` alone. The manual's ex_2 also writes the identifier `Id_1` as `id_1`, which the User Manual allows (regular names are case insensitive); this example writes the names as declared. `partition except all` leaves no identifier to partition by, so the whole data set is one partition (the rule for an omitted partition clause does not apply: the clause is given). The rank page: "If some Data Points are in the same order according to the specified orderClause, the same order number (rank) is assigned and a gap appears in the sequence of the assigned ranks".

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

The whole data set in the order of `Id_2`: 2020Q1 (`A` and `B`) → rank 1 (tie), 2020Q2 (`A` and `B`) → rank 3 (tie), 2021Q1 (`A`) → rank 5, 2021Q2 (`A`) → rank 6.
