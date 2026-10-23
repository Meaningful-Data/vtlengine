# `min` as an analytic function in `calc`

Authored: the manual has no example of this form. The Analytic invocation says: "For the invocation at Component level through the calc operator or the calc clause, the resulting Data Set retains all the original components of the input Data Set plus the components explicitly calculated by it". With no order and no window clause the window is the whole partition of `Id_1`.

Input: the chapter's shared `DS_1` (`Id_1` String and `Id_2` Time_Period identifiers, `Me_1` and `Me_2` Number measures, no null):

| Id_1 | Id_2 | Me_1 | Me_2 |
|---|---|---|---|
| A | 2020Q1 | 3 | 1.5 |
| A | 2020Q2 | 1 | 3.5 |
| A | 2021Q1 | 5 | 2.5 |
| A | 2021Q2 | 1 | 0.5 |
| B | 2020Q1 | 4 | 6.5 |
| B | 2020Q2 | 8 | 4.5 |

Numbers that are not whole are written with 6 decimals.

`min` returns the smallest value; the result has the type of the operand (Number) ([Minimun value](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Minimun%20value/index.html)).

Derivation (per partition):

| Id_1 | Me_1 values | Me_3 |
|---|---|---|
| A | 3, 1, 5, 1 | 1 |
| B | 4, 8 | 4 |
