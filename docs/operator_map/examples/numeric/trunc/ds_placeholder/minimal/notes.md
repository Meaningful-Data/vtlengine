Expected derived by hand from `Numeric operators/Truncation`. The placeholder `_` stands for an
omitted `numDigit`, so the result is the same as for `trunc(DS_1)`: 3, 2 and 12345, with type
Integer (the page's own example is `trunc ( 12345.6, _ )` gives `12345`).

The measure keeps its name, `Me_1`. The page declares the result as
`dataset { measure<number> _+ }` and defers to the typical behaviour of the unary operators,
which applies the operator to every measure "in order to produce homonymous Measures in DS_r".
The manual's own Ceiling and Floor `ex_1` turn two number measures, `Me_1` and `Me_2`, into
integers on a two-measure dataset, and their results are homonymous: so the manual does not
treat a number that becomes an integer as a change of the data type that renames the measure.
The pages that do rename say so: String length and String pattern location type their result
as `dataset { measure<integer[value >= 0]> int_var }`. The section "Operators changing the data
type" would also give `int_var` (mono-measure datasets only), but nothing puts this operator
among them.
The variant `ds_no_digits` has the same operator on two measures (`minimal_2`).
