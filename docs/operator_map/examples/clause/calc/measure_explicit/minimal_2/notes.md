# calc measure, change of role of an existing Attribute

The manual lets `calc` change the role of a component:

> It is possible to specify the role of the calculated Component among **measure**,
> **identifier**, **attribute**, or **viral attribute**, therefore the calc clause can be used
> also to change the role of a Component when possible.

`At_1` is an Attribute of `DS_1`; `calc measure At_1 := At_1` recalculates it with its own
values and the role Measure. The component keeps its name (`At_1`), so the result has two
Measures, `Me_1` and `At_1`, with unchanged values.
