Expected derived by hand from `Numeric operators/Truncation`. With `numDigit` omitted the
measures become integers and the decimals are dropped: 3.14159 gives 3, 2.71828 gives 2 and
12345.6 gives 12345 (the page's own example is `trunc ( 12345.6 )` gives `12345`). The result
type is Integer.

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
