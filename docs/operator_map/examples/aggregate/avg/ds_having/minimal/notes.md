# `avg` with `group by` and `having`

Authored: the manual has no example of this form. The Aggregate invocation says that the havingCondition "must invoke aggregate operators" on the groups, and "only groups for which havingCondition evaluates to true appear in the result". The condition references only an aggregated expression of a measure of `DS_1` (`Me_1`, which the data set form aggregates), so the manual and the reading of #1119 (having references identifiers and aggregated expressions only) agree.

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

`avg` returns the average of the values (sum / n); the result is always a number ([Average value](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Average%20value/index.html)).

The condition `avg ( Me_1 ) > 3` per group of `Id_1`: `A`: 2.500000, `B`: 6. Only `B` passes.

Derivation of the kept groups:

| Id_1 | Me_1 values | Me_2 values | result Me_1 | result Me_2 |
|---|---|---|---|---|
| B | 4, 8 | 6.5, 4.5 | 6 | 5.500000 |
