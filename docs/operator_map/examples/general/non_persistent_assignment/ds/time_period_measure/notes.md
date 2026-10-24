Two quarters and a null.

`DS_r := DS_1` is a non-persistent assignment: "The result re is a non-persistent Data Set that has
the same data structure as the Operand" (Non-persistent assignment, Behaviour). So the result has
the structure and the data points of `DS_1`: every value comes back as it went in, null included,
and the expected structure states the nullability of the components, `Id_1` not nullable and `Me_1`
nullable.

| Id_1 | Me_1 |
|---|---|
| 1 | 2020Q1 |
| 2 | 2020Q2 |
| 3 | null |
