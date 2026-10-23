# calc items are independent

The manual:

> The *calcExpr* sub-expressions are independent one another, they can only reference
> Components of the input Data Set and cannot use Components generated, for example, by other
> *calcExpr*.

So `Me_2 := Me_1 * 10` reads the `Me_1` of the input (10, 20), not the value that the first
item calculates (11, 21):

| Id_1 | input Me_1 | Me_1 := Me_1 + 1 | Me_2 := Me_1 * 10 |
|---|---|---|---|
| 1 | 10 | 11 | 100 |
| 2 | 20 | 21 | 200 |

A sequential evaluation would give `Me_2` = 110 and 210.
