Expected derived by hand from `Numeric operators/Truncation`: `numDigit` is
`component<integer> | integer`, so it may be a component while the operand is a scalar.
Me_2 = trunc(3.14159, Me_1): with 2 digits the decimals after the second are dropped, so 3.14;
with 4, 3.1415; with 0, 3.0. With `numDigit` given the result is a Number. Me_1 is kept.
