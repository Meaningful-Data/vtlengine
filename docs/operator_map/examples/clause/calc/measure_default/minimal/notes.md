# calc, role omitted, new component

`Me_2` does not exist in `DS_1` and no role is written, so it is a Measure. The manual:

> When the role is omitted, the following rule is applied: if the component exists in the
> operand Data Set then it maintains its role; if the component does not exist in the operand
> Data Set then its role is Measure.

> If the calculated Component is a new Component, it is added to the output Data Set.

So the result keeps `Id_1` and `Me_1` and adds the Measure `Me_2 = Me_1 * 2` (an Integer, as
`Me_1 * 2` is in the manual's own example `ex_1`). Manual example `ex_1` covers the other half
of the rule (an existing Measure keeps its role), `minimal_2` an existing Attribute.
