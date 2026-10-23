# aggr with the role attribute

`group by Id_1` groups the data points by `Id_1` and drops `Id_2`. Two aggregated components
are calculated per group: the Measure `Me_1 = sum(Me_1)` (no role written, and `Me_1` is a
Measure of `DS_1`) and the new component `At_1 = max(Me_1)` with the role `attribute`. The
manual: "It is possible to specify the role of the calculated Component among **measure**,
**attribute**, or **viral attribute**" and "The Attributes calculated through the **aggr**
clauses are maintained in the result".

| Id_1 | Me_1 values | sum | max |
|---|---|---|---|
| 1 | 1, 5 | 6 | 5 |
| 2 | 3, 4 | 7 | 4 |

The `Me_1 := sum(Me_1)` item keeps a Measure in the result: "The **aggr** computed Measures and
Attributes are the only Measures and Attributes returned in the output Data Set", so without it
the result would have no Measure. That form is valid too (User Manual: "A Data Set without
Measures is allowed") and is the example `minimal_2`.

Type of the `sum`: the Sum page declares the result **number** and adds "Since **integer** is a
subtype of **number**, implementations may preserve the **integer** type for the resulting
measure or component when all operand values are **integer** and the sum is computed in
integer arithmetic", so the expected `Me_1` may be Integer or Number.

`max` keeps the type of its operand (Max page: "the result has the **same** type as the operand"),
so `At_1` is an Integer.
