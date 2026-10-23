# aggr ... group all time_agg("A")

The Aggregation page: **group all** "converts the values of an Identifier Component using
*conversionExpr* and keeps all the resulting Identifiers". The Time aggregation page says that
inside an aggregation `time_agg` is written without its operand, and that the data set must
have exactly one time Identifier (`Id_1`, a quarterly `time_period` here); on `time_period`
"the original period indicator is converted in the target one".

So `Id_1` becomes annual and the data points are grouped by the converted `Id_1` and `Id_2`:

| converted Id_1 | Id_2 | data points | sum(Me_1) |
|---|---|---|---|
| 2020A | A | 2020Q1 (1), 2020Q2 (2) | 3 |
| 2021A | A | 2021Q1 (4) | 4 |
| 2020A | B | 2020Q1 (8) | 8 |

The Aggregate invocation page marks `group all` as deprecated for the aggregate operators; the
Aggregation page of the clause does not.

Type of the `sum`: the Sum page declares the result **number** and adds "Since **integer** is a
subtype of **number**, implementations may preserve the **integer** type for the resulting
measure or component when all operand values are **integer** and the sum is computed in
integer arithmetic", so the expected `Me_1` may be Integer or Number.
