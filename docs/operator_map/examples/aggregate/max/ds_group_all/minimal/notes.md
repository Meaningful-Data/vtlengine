# `max` with `group all time_agg ( "A" )`

Authored: the manual has no example of this (deprecated) form. The Aggregate invocation says: "group all: converts the values of an Identifier Component using conversionExpr and keeps all the resulting Identifiers", and `time_agg` says: "On time_period type, the operator maps the input value into the comprising larger time period specified by the periodIndTo parameter". `time_agg ( "A" )` applies to the only time identifier, `Id_2`: each quarter becomes its year, and the data points are grouped by `Id_1` and that year. The annual periods are written `2020` and `2021`.

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

`max` returns the greatest value; the result has the type of the operand (Number) ([Maximum value](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Maximum%20value/index.html)).

Derivation:

| Id_1 | Id_2 (annual) | Me_1 values | Me_2 values | result Me_1 | result Me_2 |
|---|---|---|---|---|---|
| A | 2020 | 3, 1 | 1.5, 3.5 | 3 | 3.500000 |
| A | 2021 | 5, 1 | 2.5, 0.5 | 5 | 2.500000 |
| B | 2020 | 4, 8 | 6.5, 4.5 | 8 | 6.500000 |
