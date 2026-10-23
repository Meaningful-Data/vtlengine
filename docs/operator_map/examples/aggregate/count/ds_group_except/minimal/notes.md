# `count` with `group except`

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

`count` returns the number of Data Points ([Counting the number of data points](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Counting%20the%20number%20of%20data%20points/index.html)).

Team decision [#1049](https://github.com/Meaningful-Data/vtlengine/issues/1049) (📌): `count` over a Data Set always returns a single Integer Measure `int_var`, whatever the Measures of the operand (the count page's result type `dataset { measure<integer> int_var }` prevails over the Aggregate invocation's "the resulting Data Set has the same Measures as the operand"), and a Data Point whose Measures are all null is ignored. `DS_1` has no null, so `int_var` is the number of Data Points of the group.

Derivation:

| Id_1 | Me_1 values | Me_2 values | result int_var |
|---|---|---|---|
| A | 3, 1, 5, 1 | 1.5, 3.5, 2.5, 0.5 | 4 |
| B | 4, 8 | 6.5, 4.5 | 2 |
