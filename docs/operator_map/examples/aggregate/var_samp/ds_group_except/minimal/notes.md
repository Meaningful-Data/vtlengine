# `var_samp` with `group except`

Authored: the manual has no example of this form. The Aggregate invocation says: "group except: the Data Points are grouped by the values of the Identifiers not specified as groupingId. The Identifiers specified as groupingId are dropped in the result." So `group except Id_2` groups by `Id_1` and drops `Id_2`.

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

`var_samp` returns the sample variance, sum((x - mean)^2) / (n - 1) (the manual's ex_1 gives 4.0 for 3, 5, 7) ([Sample variance](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Sample%20variance/index.html)).

Derivation:

| Id_1 | Me_1 values | Me_2 values | result Me_1 | result Me_2 |
|---|---|---|---|---|
| A | 3, 1, 5, 1 | 1.5, 3.5, 2.5, 0.5 | 3.666667 | 1.666667 |
| B | 4, 8 | 6.5, 4.5 | 8 | 2 |
