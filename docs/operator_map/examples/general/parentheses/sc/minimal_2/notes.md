# Nested parentheses on scalars

Authored from the text of [Parentheses](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Parentheses/index.html): "If more parentheses are
nested, the inner parentheses are evaluated first, for example `( 20 - 10 / (2 + 3) ) * 3` would
give `54`."

Derivation: `2 + 3` = 5, then `10 / 5` = 2, then `20 - 2` = 18, then `18 * 3` = 54. The division
"has type *number*" (Division page), and an operation with a number operand gives a number
(Subtraction and Multiplication pages), so the result is the number 54.
