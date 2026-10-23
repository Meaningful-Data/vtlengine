# Parentheses at component level

Authored: the manual's only example of
[Parentheses](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Parentheses/index.html) is at dataset level. Its syntax examples include
`( CMP_1 - CMP_2 )`, and the operand type is "dataset | component | scalar".

"The operations enclosed in the parentheses are evaluated first": `Me_1 + 2` is computed, then
multiplied by 3. Without the parentheses, the multiplication would come first and give
`Me_1 + 6` (7 and 10), so the example tells the two orders apart. Integer operands give an
integer result.

| Id_1 | Me_1 | (Me_1 + 2) * 3 | Me_2 |
|---|---|---|---|
| 1 | 1 | 3 * 3 | 9 |
| 2 | 4 | 6 * 3 | 18 |
