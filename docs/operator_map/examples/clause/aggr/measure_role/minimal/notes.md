# aggr with the role measure

`aggrRole ::= measure | attribute | viral attribute`. The role `measure` is written for the new
component `Me_2`, which is the sum of `Me_1` per `Id_1` group: 1 + 5 = 6 for `Id_1 = 1` and
3 + 4 = 7 for `Id_1 = 2`. `group by Id_1` drops `Id_2`. "The **aggr** computed Measures and
Attributes are the only Measures and Attributes returned in the output Data Set", so `Me_1` is
not in the result.

Type of the `sum`: the Sum page declares the result **number** and adds "Since **integer** is a
subtype of **number**, implementations may preserve the **integer** type for the resulting
measure or component when all operand values are **integer** and the sum is computed in
integer arithmetic", so the expected `Me_2` may be Integer or Number.
