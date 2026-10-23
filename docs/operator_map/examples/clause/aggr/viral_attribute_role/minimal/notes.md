# aggr with the role viral attribute

The same aggregation as `attribute_role/minimal`, with the role `viral attribute` for `At_1`:
the result holds the Viral Attribute `At_1 = max(Me_1)` per `Id_1` group (5 and 4) next to
`Me_1 = sum(Me_1)` (6 and 7). The manual: "The Attributes calculated through the **aggr**
clauses are maintained in the result. For all the other Attributes that are defined as
**viral**, the Attribute propagation rule is applied". `DS_1` has no other Attribute, so no
propagation rule is involved.

Why the `Me_1 := sum(Me_1)` item is there: the reason given in `attribute_role/minimal`. It is a
precaution so that the result keeps a Measure, not a workaround for an engine failure.
`minimal_2` holds the attribute-only form, which the User Manual allows ("A Data Set without
Measures is allowed").

Type of the `sum`: the Sum page declares the result **number** and adds "Since **integer** is a subtype of **number**, implementations may preserve the **integer** type for the resulting measure or component when all operand values are **integer**", so the expected `Me_1` may be Integer or Number. `At_1` (`max` over an Integer) is an Integer.
