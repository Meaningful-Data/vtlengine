# `max(DS_1)#Me_1`: membership on a dataset without identifiers

Manual example 7 of [Membership](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html).
The manual's expected output was re-derived and agrees.

`max(DS_1)` aggregates every data point into one, with no identifiers: `Me_1` = max(1, 2, 3) = 3
and `Me_2` = max(5, 10, 12) = 12 (the non-viral attribute `At_1` is dropped). The page says: "If
ds has no Identifier Components (for example is a result of an aggregate invocation), comp must
be a Data Set Component of the Data Set ds (a Measure or an Attribute). In this case the operator
returns a scalar value equal to the value of `comp` in `ds`". So the result is the integer 3.
