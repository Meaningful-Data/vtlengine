# valuedomain_rule: `hierarchy ( DS_1, HR_1 rule Id_2 )`

Authored, for the manual page [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html).

The ruleset is defined on the value domain `VD_1` (`value_domains/VD_1.json`, the codes A to E),
so the call names the identifier of the operand that takes its values: `rule Id_2` ("If *hr* is
defined on Value Domains then it is mandatory to specify the *condition* (if any) and the *rule*
parameters", page, "Additional Constraints"). Mode, input and output take their defaults:
`non_null`, `rule`, `computed`.

## Derivation

- C = D + E = 2 + 3 = 5.
- A = B + C, with C taken from the rule that computes it (input `rule`): 1 + 5 = 6.

Both are not null, so `non_null` returns both; output `computed` leaves out the input data
points: `(1, A, 6)` and `(1, C, 5)`.
