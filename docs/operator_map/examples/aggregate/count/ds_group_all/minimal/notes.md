# `count` with `group all time_agg ( "A" )`

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

`count` returns the number of Data Points ([Counting the number of data points](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Counting%20the%20number%20of%20data%20points/index.html)).

Team decision [#1049](https://github.com/Meaningful-Data/vtlengine/issues/1049) (📌): `count` over a Data Set always returns a single Integer Measure `int_var`, whatever the Measures of the operand (the count page's result type `dataset { measure<integer> int_var }` prevails over the Aggregate invocation's "the resulting Data Set has the same Measures as the operand"), and a Data Point whose Measures are all null is ignored. `DS_1` has no null, so `int_var` is the number of Data Points of the group.

Derivation:

| Id_1 | Id_2 (annual) | Me_1 values | Me_2 values | result int_var |
|---|---|---|---|---|
| A | 2020 | 3, 1 | 1.5, 3.5 | 2 |
| A | 2021 | 5, 1 | 2.5, 0.5 | 2 |
| B | 2020 | 4, 8 | 6.5, 4.5 | 2 |
