Expected derived by hand from `Numeric operators/Rounding`, on a dataset with two measures. With
`numDigit` omitted the measures become integers. Me_1: 3.14159 gives 3 (first decimal 1),
2.71828 gives 3 (first decimal 7), 7.2 gives 7. Me_2: 12345.6 gives 12346 (first decimal 6),
0.4 gives 0, 5.9 gives 6.

Both measures keep their names, `Me_1` and `Me_2`, with type Integer: the page declares the
result as `dataset { measure<number> _+ }` and defers to the typical behaviour of the unary
operators, which applies the operator to every measure "in order to produce homonymous Measures
in DS_r". The manual's Ceiling and Floor `ex_1` are the same case, two number measures that
become integers, and their results are homonymous; so the manual does not treat a number that
becomes an integer as a change of the data type that renames the measure. The section
"Operators changing the data type" allows such operators on mono-measure datasets only, and
nothing puts this operator among them. `minimal` is the same operator on one measure.
