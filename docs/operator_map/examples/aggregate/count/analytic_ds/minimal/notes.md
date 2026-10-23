# `count` as an analytic function at data set level

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

`count` returns the number of Data Points ([Counting the number of data points](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Counting%20the%20number%20of%20data%20points/index.html)).

The count page gives the result type `dataset { measure<integer> int_var }` for the data set form, against the Analytic invocation's "the resulting Data Set has the same Identifiers and Measures as the input Data Set": this is the same conflict that #1049 settles for the aggregate invocation, and its reasoning (the specific result type of count prevails) is followed here.

Team decision [#1049](https://github.com/Meaningful-Data/vtlengine/issues/1049) (📌): `count` over a Data Set always returns a single Integer Measure `int_var`, whatever the Measures of the operand (the count page's result type `dataset { measure<integer> int_var }` prevails over the Aggregate invocation's "the resulting Data Set has the same Measures as the operand"), and a Data Point whose Measures are all null is ignored. `DS_1` has no null, so `int_var` is the number of Data Points of the group.

Derivation (per partition):

| Id_1 | Me_1 values | Me_2 values | result int_var |
|---|---|---|---|
| A | 3, 1, 5, 1 | 1.5, 3.5, 2.5, 0.5 | 4 |
| B | 4, 8 | 6.5, 4.5 | 2 |
