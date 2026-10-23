Manual example `Numeric operators/Random/ex_2`, with its illustrative values left out.

The manual prints `Me_2` as 0.7545341, 0.3457166 and 0.5183224, but only to illustrate the
result: a random sequence cannot be fixed, and the page says only that the operator "returns a
random decimal number >= 0 and <1". No engine can be expected to give those numbers, so
`Me_2` is blank in `expected/DS_r.csv` (the adaptation) and the example is checked by property.

`calc` keeps the seed `Me_1` (16.0, 4.0, 7.2), and the expected file has it. The check compares
the columns that have values in the expected file, here the identifiers and `Me_1`, as an exact
example does, and takes the columns that are empty in it for draws: the values of `Me_2` must be
numbers in [0, 1), the same on every run, and the two engines must still give the same ones.
The authored example `minimal` checks the same form with a seed that `calc` overwrites.
