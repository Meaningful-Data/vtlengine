# check_hierarchy: rule

Manual page: [Check hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html).

`HR_2` is defined on the value domain `VD_1`, so `rule Id_2` is mandatory: it binds the rule
value domain to the component `Id_2`. `value_domains/VD_1.json` is the `VD_1` of the manual's
example 1 (the 26 uppercase letters). Mode, input and output are the defaults: `non_null`,
`dataset`, `invalid`. This example isolates the `rule` parameter on a small ruleset; the
manual's example 1 is the other example of the variant. The rules give no errorcode or
errorlevel, so both are null.

- R1 `A = B + C`: `10 = 3 + 6` is false, imbalance 10 - 9 = 1;
- R2 `C >= B`: `6 >= 3` is true, so it is not in the `invalid` output.

`Me_1` is Number, so `imbalance` is Number. Subtraction: "If one of the operands is of type
*number*, then the other operand is implicitly cast to *number* and therefore the result has
type *number*".

`errorlevel` (null here) may be Integer or Number: "VTL assumes that a Value Domain
errorlevel_vd of the error levels exists in the Information Model" (define hierarchical
ruleset), so its type is the Information Model's choice; the `errorlevel` variant tests it.
