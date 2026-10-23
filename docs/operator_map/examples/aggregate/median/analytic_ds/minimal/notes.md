# `median` as an analytic function at data set level

Authored: the manual has no example of this form. The Analytic invocation says: "For each input Data Point, one output Data Point is returned, having the same values of the Identifiers", and "If windowClause is not specified, then the set of Data Points which contribute to the analytic operation is the whole partition which the current Data Point belongs to". There is no order clause, so the two defaults the manual gives for an omitted window agree: the whole partition. Every data point gets the value of its partition of `Id_1`.

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

Derivation (per partition):

| Id_1 | Me_1 values | Me_2 values | result Me_1 | result Me_2 |
|---|---|---|---|---|
| A | 3, 1, 5, 1 | 1.5, 3.5, 2.5, 0.5 | 2 | 2 |
| B | 4, 8 | 6.5, 4.5 | 6 | 5.500000 |
