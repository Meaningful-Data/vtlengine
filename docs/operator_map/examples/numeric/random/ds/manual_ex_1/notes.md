Manual example `Numeric operators/Random/ex_1`, with its illustrative values left out.

The manual prints `Me_1` as 0.3582791, 0.428819 and 0.715488, but only to illustrate the
result: a random sequence cannot be fixed, and the page says only that the operator "returns a
random decimal number >= 0 and <1". No engine can be expected to give those numbers, so `Me_1`
is blank in `expected/DS_r.csv` (the adaptation) and the example is checked by property, as the
design says for `random`: the identifiers are the manual's, the values of `Me_1` must be
numbers in [0, 1), the same on every run, and the two engines must still give the same ones.
