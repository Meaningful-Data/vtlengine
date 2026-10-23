# Counting, ex_1: `count ( DS_1 group by Id_1 )`

The expected output is the manual's, a single Integer measure `int_var`. It also follows the
team decision [#1049](https://github.com/Meaningful-Data/vtlengine/issues/1049) (📌): `count`
over a Data Set always returns the single measure `int_var` (the count page's result type
`dataset { measure<integer> int_var }` prevails over the Aggregate invocation's "the resulting
Data Set has the same Measures as the operand", which would keep `Me_1`), and a Data Point
whose Measures are all null is ignored. `DS_1` has no null: 2011 → 3, 2012 → 2.
