Manual example `Numeric operators/Random/ex_2`, imported as it is.

The manual's values (0.7545341, 0.3457166, 0.5183224) only illustrate the result: a random
sequence cannot be fixed, and the page says only that the operator "returns a random decimal
number >= 0 and <1". So the expected values of `Me_2` are not compared. The example is checked
by property, as the design says for `random`: the values are numbers in [0, 1), the same on
every run, and the two engines must give the same values.

The property check looks at every value that is not an identifier, and `calc` keeps `Me_1`, the
seed (16.0, 4.0, 7.2). So both engines are reported as a mismatch, `'16.0' is not a number in
[0, 1)`, although the values of the new measure `Me_2` are in [0, 1) and stable on both. The
check cannot tell the kept column from the random one. The example is 🔵 anyway, because the
engines give different values.
