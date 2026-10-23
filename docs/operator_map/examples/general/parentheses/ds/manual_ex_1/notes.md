# `(DS_1 + DS_2) * DS_2`

Manual example 1 of [Parentheses](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Parentheses/index.html).
The manual's expected output was re-derived and agrees.

The sum in the parentheses comes first. A binary operation on datasets pairs the data points with
the same identifier values, so only `(10, A)` and `(11, B)` are in both operands (`(10, B)` and
`(11, A)` are only in `DS_1`, `(10, C)` only in `DS_2`). Then the product with `DS_2` pairs the
same two data points.

| Id_1 | Id_2 | Me_1: (DS_1 + DS_2) * DS_2 | Me_2: (DS_1 + DS_2) * DS_2 |
|---|---|---|---|
| 10 | A | (5 + 10) * 10 = 150 | (5.0 + 3.0) * 3.0 = 24.0 |
| 11 | B | (4 + 6) * 6 = 60 | (20.3 + 7.0) * 7.0 = 191.1 |

`Me_1` stays an integer and `Me_2` a number. Without the parentheses, `DS_1 + DS_2 * DS_2`
would give 5 + 10 * 10 = 105 and 5.0 + 3.0 * 3.0 = 14.0 for `(10, A)`.
