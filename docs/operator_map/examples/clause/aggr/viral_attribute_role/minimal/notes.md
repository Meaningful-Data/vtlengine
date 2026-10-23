# aggr with the role viral attribute

The same aggregation as `attribute_role/minimal`, with the role `viral attribute` for `At_1`:
the result holds the Viral Attribute `At_1 = max(Me_1)` per `Id_1` group (5 and 4) next to
`Me_1 = sum(Me_1)` (6 and 7). The manual: "The Attributes calculated through the **aggr**
clauses are maintained in the result. For all the other Attributes that are defined as
**viral**, the Attribute propagation rule is applied". `DS_1` has no other Attribute, so no
propagation rule is involved.

The `Me_1 := sum(Me_1)` item keeps a Measure in the result, as in `attribute_role/minimal`. The
form without it, a result with no Measure (User Manual: "A Data Set without Measures is
allowed"), is the example `minimal_2`.

Type of the `sum`: the Sum page declares the result **number** and adds "Since **integer** is a
subtype of **number**, implementations may preserve the **integer** type for the resulting
measure or component when all operand values are **integer** and the sum is computed in
integer arithmetic", so the expected `Me_1` may be Integer or Number.

`max` keeps the type of its operand (Max page: "the result has the **same** type as the operand"),
so `At_1` is an Integer.
