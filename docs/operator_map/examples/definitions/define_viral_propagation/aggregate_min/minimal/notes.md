# define viral propagation: `aggregate min`

The rule is an aggregate one (`aggregationClause ::= aggregate aggregateFunction`):
"The rule simply names the aggregate function that must collapse the set of operand values to a
single result", and "The semantics of these aggregates follow the standard VTL definitions".
`At_1` is a Number viral Attribute, so the aggregate applies to numbers.

`DS_1 + DS_2` pairs the data points with the same `Id_1` and adds their Measures ([Typical
behaviours](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html), operators on two Data Sets). The viral Attribute `At_1` of each pair is
"calculated by applying the proper Attribute propagation algorithm on the values of the
Attributes of DP_1 and DP_2", and it stays viral in the result ("DS_r maintains the Attributes
declared as viral for the operand Data Sets; these Attributes are considered as viral also in
DS_r").

`min` of each pair is the smaller value:

| Id_1 | DS_1 At_1 | DS_2 At_1 | DS_r At_1 |
|---|---|---|---|
| 1 | 1.5 | 4.0 | 1.5 |
| 2 | 3.0 | 2.0 | 2.0 |
