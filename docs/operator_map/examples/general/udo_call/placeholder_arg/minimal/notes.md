# Call with `_` for an optional argument

Authored. [User-defined operator call](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/User-defined%20operator%20call/index.html): "An underscore (`_`) can be used to denote
that the value for an optional operand is omitted." The `_` stands in the middle, where an
argument cannot simply be left out, because `z` comes after it.

So `x` = 10, `y` takes its default 1 and `z` = 5: `x - y - z` = 10 - 1 - 5 = 4, an integer.
Reading the 5 as `y` (with `z` at its default 100) would give -95.
