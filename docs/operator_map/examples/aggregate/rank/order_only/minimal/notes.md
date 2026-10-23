# `rank` with an order clause and no partition clause

Authored: the manual's only example has a partition clause. The Analytic invocation says of the partition clause: "If the clause is omitted, then the Data Set is partitioned by the Identifier Components that are not specified in the orderClause." The order clause names `Id_2` only, so the data set is partitioned by `Id_1`. The rank page: "The operator returns an order number (rank) for each Data Point, starting from the number 1 and following the order specified in the orderClause".

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

Partition `A` in the order of `Id_2`: 2020Q1 → 1, 2020Q2 → 2, 2021Q1 → 3, 2021Q2 → 4; partition `B`: 2020Q1 → 1, 2020Q2 → 2. (Ranking the whole data set instead, as SQL would without a PARTITION BY, would give 1, 3, 5, 6, 1, 3: the manual's rule is what this example checks.)
