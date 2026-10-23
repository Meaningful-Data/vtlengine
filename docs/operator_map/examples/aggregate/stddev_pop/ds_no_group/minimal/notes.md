# `stddev_pop` over the whole data set

Authored: the manual has no example of this form. The Aggregate invocation says: "If the grouping clause is omitted, then all the input Data Points are aggregated in a single group and the clause returns a Data Set that contains a single Data Point and has no Identifiers."

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

`stddev_pop` returns the population standard deviation, sqrt(sum((x - mean)^2) / n) (the manual's ex_1 gives 1.632993 for 3, 5, 7) ([Population standard deviation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Population%20standard%20deviation/index.html)).

Derivation (one group, all six data points):

| Me_1 values | Me_2 values | result Me_1 | result Me_2 |
|---|---|---|---|
| 3, 1, 5, 1, 4, 8 | 1.5, 3.5, 2.5, 0.5, 6.5, 4.5 | 2.426703 | 1.972027 |
