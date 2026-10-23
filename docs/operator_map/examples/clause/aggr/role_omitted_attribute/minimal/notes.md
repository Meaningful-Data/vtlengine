# aggr, role omitted, existing Attribute

The Aggregation page has the same rule as `calc`:

> When the role is omitted, the following rule is applied: if the component exists in the
> operand Data Set then it maintains its role; if the component does not exist in the operand
> Data Set then its role is Measure.

Both items are written without a role. `Me_1` is a Measure of `DS_1`, so it stays a Measure.
`At_1` is an Attribute of `DS_1`, so it stays an **Attribute**, and "The Attributes calculated
through the **aggr** clauses are maintained in the result". The User Manual adds that it also
stays non-viral: "by default a non-viral Attribute in the input generates a non-viral Attribute
also in the result (this happens when the Attribute in the result is calculated through an
explicitly expression but without specifying explicitly its virality)".

Per `Id_1` group (`group by Id_1` drops `Id_2`):

| Id_1 | Me_1 values | sum(Me_1) | At_1 values | max(At_1) |
|---|---|---|---|---|
| 1 | 1, 5 | 6 | 7, 2 | 7 |
| 2 | 3, 4 | 7 | 4, 9 | 9 |

Each item aggregates its own component. The Measure item keeps the result an ordinary data set
with a Measure; its role-omitted reading is the one of the manual's own `aggr` examples
(`ex_1`: `Me_1 := sum(Me_1)`). The attribute-only form is covered by `attribute_role/minimal_2`.

Type of the `sum`: the Sum page declares the result **number** and adds "Since **integer** is a
subtype of **number**, implementations may preserve the **integer** type for the resulting
measure or component when all operand values are **integer** and the sum is computed in
integer arithmetic", so the expected `Me_1` may be Integer or Number.

`max` keeps the type of its operand (Max page: "the result has the **same** type as the operand"),
so `At_1` is an Integer.

Triage (2026-10-23): both engines return `At_1` as a Measure, the same finding as
`calc/measure_default/minimal_2`. Re-derived from the rule quoted above, the expected output
stands: an existing component keeps its role when `aggr` omits it. No erratum and no team
decision applies.
