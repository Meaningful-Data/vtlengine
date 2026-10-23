# condition: `hierarchy ( DS_1, HR_1 condition Id_1 )`

Authored, for the manual page [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html).

The ruleset is defined on the variable `Id_2`, with the conditioning variable `Id_1`, and its only
rule applies where `Id_1 = 1`. For a ruleset defined on variables the components of the
`condition` and `rule` clauses may be written in the call, and then they must be the ones of the
ruleset's signature, in the same order (page, "Additional Constraints"); here only the
`condition` clause is written. Mode, input and output take their defaults: `non_null`, `rule`,
`computed`.

## Derivation

- `Id_1 = 1`: the condition is true, so the rule is evaluated ("it is evaluated only when the
  leftCondition is **true**", `define hierarchical ruleset`): C = D + E = 1 + 2 = 3. It is not
  null, so `non_null` returns it.
- `Id_1 = 2`: the condition is false, so the rule gives no data point.

Output `computed` returns only the computed data points: `(1, C, 3)`.
