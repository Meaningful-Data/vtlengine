Expected derived by hand from `Numeric operators/Floor`, on a dataset with one measure: the
greatest integer smaller than or equal to the value. 7.2 gives 7, -3.7 gives -4 and 5.0 gives 5.

The measure keeps its name, `Me_1`, with type Integer: the page declares the result as
`dataset { measure<integer> _+ }`, and the manual's own `ex_1` of this page turns two number
measures, `Me_1` and `Me_2`, into integers with homonymous results. The pages that do rename say
so: String length and String pattern location type their result as
`dataset { measure<integer[value >= 0]> int_var }`. The section "Operators changing the data
type" would give `int_var` here (mono-measure only), but nothing puts this operator among them.
This example has one measure, where `ex_1` has two.
