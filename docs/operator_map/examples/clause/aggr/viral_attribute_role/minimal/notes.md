# aggr with the role viral attribute

The same aggregation as `attribute_role/minimal`, with the role `viral attribute` for `At_1`:
the result holds the Viral Attribute `At_1 = max(Me_1)` per `Id_1` group (5 and 4) next to
`Me_1 = sum(Me_1)` (6 and 7). The manual: "The Attributes calculated through the **aggr**
clauses are maintained in the result. For all the other Attributes that are defined as
**viral**, the Attribute propagation rule is applied". `DS_1` has no other Attribute, so no
propagation rule is involved.
