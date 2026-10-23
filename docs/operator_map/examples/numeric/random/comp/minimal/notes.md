Authored example of the component form `ds [calc r := random(col_1, 12)]` of `Numeric
operators/Random`. `calc Me_1 := random(Me_1, 5)` replaces the seed measure by the draws (the
typical behaviour of the operators on components: a result component that is one of the original
measures replaces its values). So the only value of the result that is not an identifier is a
draw, and the property check, which looks at every such value, can be used.

The values are not fixed: the page says only that the operator "returns a random decimal number
>= 0 and <1". The example is checked by property, as the design says for `random`: the values
are numbers in [0, 1), the same on every run, and the two engines must give the same values.
The expected output therefore states the structure only (`Id_1` an Integer identifier, `Me_1` a
Number measure) and has no data points.

The manual's own example of this form, `manual_ex_2`, calculates a new measure and keeps the
seed column, which the property check cannot tell apart from the draws.
