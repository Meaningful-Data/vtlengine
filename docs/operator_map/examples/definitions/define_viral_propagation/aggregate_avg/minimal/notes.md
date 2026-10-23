# define viral propagation: `aggregate avg`

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

`avg` of each pair is the mean:

| Id_1 | DS_1 At_1 | DS_2 At_1 | DS_r At_1 |
|---|---|---|---|
| 1 | 1.5 | 4.0 | 2.75 |
| 2 | 3.0 | 2.0 | 2.5 |

Issue #1175 (viral `avg` and the associative law) was closed on 2026-10-22 with a team reading: `avg` relies on the order of execution across operators and takes the whole set within one operator (the clarification proposed in sdmx-twg/vtl#744). Over a single pair both readings give the mean, so this example follows the manual and needs no decision marker; `minimal_2` and `minimal_3` show the two cases where the reading decides the result. `At_1` is a Number, so no question of a type change from an integer mean arises.
