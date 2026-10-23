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
