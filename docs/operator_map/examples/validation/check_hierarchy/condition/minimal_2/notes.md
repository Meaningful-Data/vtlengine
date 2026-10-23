# check_hierarchy: condition (ruleset on value domains)

Manual pages: [Check hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html) and define hierarchical
ruleset.

The case where `condition` matters. `HR_3` has the rule of `HR_2` (the `condition/minimal`
example) written on value domains: the conditioning value domain `VD_C` and the rule value
domain `VD_1`. "If *hr* is defined on Value Domains then it is mandatory to specify the
*condition* (if any in the ruleset *hr*) and the *rule* parameters. Moreover, the Components
specified as *condComp* and *ruleComp* must belong to the operand *op* and must take values on
the Value Domains corresponding, in positional order, to the ones specified in the condition and
rule parameter of *hr*." `condition Id_1 rule Id_2` binds `VD_C` to `Id_1` and `VD_1` to `Id_2`.
Mode, input and output are the defaults: `non_null`, `dataset`, `invalid`.

The leftCondition `VD_C = 2` is evaluated on `Id_1`:

- `Id_1` = 1: the condition is false, the rule is not evaluated (A = 5, B = 2 would be false);
- `Id_1` = 2: `7 = 4` is false, imbalance 7 - 4 = 3.

The expected output is the one of `condition/minimal`: the false data point with `Me_1` = 7 (the
left item A). The rule gives no errorcode or errorlevel, so both are null.

`value_domains/VD_1.json` is the `VD_1` of the manual's example 1 (the 26 uppercase letters) and
`value_domains/VD_C.json` the values of `Id_1` (Integer: 1, 2).

`Me_1` is Number, so `imbalance` is Number. Subtraction: "If one of the operands is of type
*number*, then the other operand is implicitly cast to *number* and therefore the result has
type *number*".

`errorlevel` (null here) may be Integer or Number: "VTL assumes that a Value Domain
errorlevel_vd of the error levels exists in the Information Model" (define hierarchical
ruleset), so its type is the Information Model's choice; the `errorlevel` variant tests it.
