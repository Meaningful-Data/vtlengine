Expected derived by hand from `Numeric operators/Rounding`. With `numDigit` omitted the measures
become integers: 3.14159 rounds to 3 (first decimal 1), 2.71828 to 3 (first decimal 7) and
12345.6 to 12346 (first decimal 6; the page's own example is `round ( 12345.6 )` gives `12346`).
The result type is Integer.

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
`minimal_2` is the same operator on two measures.
