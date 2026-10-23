Expected derived by hand from `Numeric operators/Rounding`: `numDigit` is
`component<integer> | integer`, so it may be a component while the operand is a scalar.
Me_2 = round(3.14159, Me_1): with 2 digits the third decimal is 1, so 3.14; with 4 the fifth
decimal 9 adds 1 to the fourth, so 3.1416; with 0 the first decimal 1 leaves 3.0. With
`numDigit` given the result is a Number. Me_1 is kept.
