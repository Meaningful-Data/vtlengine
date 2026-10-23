# check: invalid

Manual page: [Check](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check/index.html).

The authored `check` examples share `DS_1` (Me_1 = 5, -3, null) and check `DS_1 > 0`, a
boolean dataset whose only measure is `bool_var`: true on 1, false on 2 and null on 3 (a
comparison with null is null).

No `imbalance` is given, so `imbalance` is null ("If not specified then imbalance is null").
Its type is the parameter's, `number`; integer is a subtype of number, so either is accepted.
errorcode and errorlevel are "the error code [level] to be produced when the condition evaluates
to false", and null when `bool_var` is null (Behaviour): they are null on true data points too.
errorlevel is integer ("or integer if the errorcode_vd Value Domain is not found").

Output `invalid`: "only the Data Points where bool_var is false are returned", so only 2;
`bool_var` stays in the structure (the Behaviour lists it for both outputs).
