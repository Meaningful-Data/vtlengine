`Numeric operators/Random`: `random(15, 12)` is one of the valid syntaxes, and the result is "a
random decimal number >= 0 and <1". The value is not fixed, so the example is checked by
property (a Number in [0, 1), the same on every run, and the same on both engines) and the
expected value is null.
