# `max` with `group by` over a String measure

Authored: this example tests `max` over a String measure (the operator takes any
comparable type). The input has the shared identifiers and a String measure:

| Id_1 | Id_2 | Me_1 |
|---|---|---|
| A | 2020Q1 | kiwi |
| A | 2020Q2 | apple |
| A | 2021Q1 | pear |
| A | 2021Q2 | fig |
| B | 2020Q1 | plum |
| B | 2020Q2 | date |

The page ([Maximum value](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Maximum%20value/index.html)) says the operator returns the maximum
of the input values, and "For other comparable scalar types (for example time, string,
boolean), the result has the same type as the operand": the result is a String. Strings compare
character by character (the Greater than page: `"hello" > "hi"` gives false). The values are
lowercase ASCII words with different first letters, so any such order gives the same result:
apple < fig < kiwi < pear and date < plum.

The greatest value is `pear` for `A` and `plum` for `B`. In `A` it is neither the first nor the last data
point in the order of `Id_2`.
