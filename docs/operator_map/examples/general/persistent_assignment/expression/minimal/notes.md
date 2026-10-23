# Persistent assignment of an expression

Authored from the syntax example `DS_r <- DS_1 - DS_2` of
[Persistent assignment](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Persistent%20assignment/index.html): "the operand `op` can be
obtained through an expression as complex as needed (for example `op` can be the expression
`DS_1 - DS_2`)". `DS_r` "assumes the same value as `op`" and "has the same data structure as the
Operand".

The two inputs have the same identifier values, so the subtraction pairs every data point:
10 - 3 = 7 and 20 - 5 = 15. Two integers give an integer (Subtraction page).
