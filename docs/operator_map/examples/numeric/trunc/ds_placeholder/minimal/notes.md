Expected derived by hand from `Numeric operators/Truncation`. The placeholder `_` stands for an
omitted `numDigit`, so the result is the same as for `trunc(DS_1)`: 3, 2 and 12345, with type
Integer (the page's own example is `trunc ( 12345.6, _ )` gives `12345`).

The measure keeps its name, `Me_1`. The page declares the result as
`dataset { measure<number> _+ }` and defers to the typical behaviour of the unary operators,
which applies the operator to every measure "in order to produce homonymous Measures in DS_r".
The pages of `length` and `instr` name a single `int_var` instead. The manual's section
"Operators changing the data type" would also give `int_var` on a mono-measure dataset, but the
page does not put this operator among them, and the manual's own ceiling and floor examples,
which turn numbers into integers as well, keep the names of the measures.
