Expected derived by hand from `Numeric operators/Rounding`. With `numDigit` omitted the measures
become integers: 3.14159 rounds to 3 (first decimal 1), 2.71828 to 3 (first decimal 7) and
12345.6 to 12346 (first decimal 6; the page's own example is `round ( 12345.6 )` gives `12346`).
The result type is Integer.

The measure keeps its name, `Me_1`. The page declares the result as
`dataset { measure<number> _+ }` and defers to the typical behaviour of the unary operators,
which applies the operator to every measure "in order to produce homonymous Measures in DS_r".
The pages of `length` and `instr` name a single `int_var` instead. The manual's section
"Operators changing the data type" would also give `int_var` on a mono-measure dataset, but the
page does not put this operator among them, and the manual's own ceiling and floor examples,
which turn numbers into integers as well, keep the names of the measures.
