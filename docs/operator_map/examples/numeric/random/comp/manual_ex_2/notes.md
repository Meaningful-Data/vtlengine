Manual example `Numeric operators/Random/ex_2`, imported as it is.

The manual's values for `Me_2` (0.7545341, 0.3457166, 0.5183224) only illustrate the result: a
random sequence cannot be fixed, and the page says only that the operator "returns a random
decimal number >= 0 and <1". No engine is expected to give these numbers.

This example is not checked by property, because it keeps the seed. The property check looks at
every value of the output that is not an identifier, and `calc` keeps `Me_1` (16.0, 4.0, 7.2),
which is not a draw: the check would report `'16.0' is not a number in [0, 1)` on both engines
although the new measure `Me_2` holds draws. So it keeps the default check, the comparison with
the manual's values, which an engine cannot pass: its result says nothing about the operator.
The status of the `random(comp, n)` variant is meant to come from the authored example
`minimal`, which overwrites the seed so that every non-identifier value is a draw, and is
checked by property.
