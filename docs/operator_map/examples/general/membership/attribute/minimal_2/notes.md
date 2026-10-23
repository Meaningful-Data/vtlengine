# `DS_1#At_1` on a dataset without identifiers

Authored from [Membership](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html): "If ds has no Identifier Components (for
example is a result of an aggregate invocation), comp must be a Data Set Component of the Data Set
ds (a Measure or an Attribute). In this case the operator returns a scalar value equal to the
value of `comp` in `ds`; given that `ds` has no Identifier Components, there can be only one value
for `comp` in `ds`."

The manual's example 7 shows the measure case (`max(DS_1)#Me_1`), and its example 8 meant the
attribute case but used a dataset with identifiers (see its erratum). Here `DS_1` has no
identifiers, so it holds a single data point, and `At_1` is an ordinary (not viral) attribute. The
result is the value of `At_1`, the string `P`.

`DS_1` is given directly, without identifiers, because an aggregation would not keep a non-viral
attribute.
