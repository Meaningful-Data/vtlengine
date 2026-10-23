# `median` at component level in `aggr`

Authored: the manual has no example of this form. The Aggregate invocation says: "For the invocation at Component level, the resulting Data Set has the Measures explicitly calculated (all the other Measures are dropped because no aggregation behaviour is specified for them)." So the result has `Id_1` and `Me_3` only.

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

`median` returns the middle value of the sorted values, or the mean of the two middle values when there is an even number of them (the manual's ex_1 gives median(2, 4) = 3.0) ([Median value](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Median%20value/index.html)).

Derivation:

| Id_1 | Me_1 values | Me_3 |
|---|---|---|
| A | 3, 1, 5, 1 | 2 |
| B | 4, 8 | 6 |
