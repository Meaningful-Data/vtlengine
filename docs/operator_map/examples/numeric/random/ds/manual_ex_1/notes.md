Manual example `Numeric operators/Random/ex_1`, imported as it is.

The manual's values (0.3582791, 0.428819, 0.715488) only illustrate the result: a random
sequence cannot be fixed, and the page says only that the operator "returns a random decimal
number >= 0 and <1". So the expected values are not compared. The example is checked by
property, as the design says for `random`: the values are numbers in [0, 1), the same on every
run, and the two engines must give the same values.
