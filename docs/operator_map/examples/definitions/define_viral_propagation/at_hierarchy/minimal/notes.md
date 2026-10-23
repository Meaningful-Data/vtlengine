# define viral propagation: propagation at a hierarchy

The combination point is the [hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html) operator: "The Attribute propagation rule is
applied on the groups of Data Points which contribute to the same Data Points of the result".
`hierarchy(DS_1, HR_1 rule Id_2)` uses the defaults: mode `non_null`, input `rule` and output
`computed` ("the resulting Data Set contains only the set of Data Points computed according to
the Ruleset"), so the result has only the computed `TOTAL` of each `Id_1`.

Enumerated rules ([define viral propagation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html)): "A binary clause (e.g., when "A" and "B"
then "C") matches if the input pair consists of exactly those two values. A unary clause (e.g.,
when "A" then "A") matches if the specified value is present in the input pair (regardless of
what the other value is). The first matching clause determines the result." The clauses "are
evaluated in order, but binary clauses (specifying a pair of values) always take precedence over
unary clauses", and "an else clause can provide the default result for any combination that is
not explicitly listed; when it is absent, missing combinations yield NULL".

| Id_1 | `TOTAL = X + Y` | At_1 of X, Y | DS_r At_1 |
|---|---|---|---|
| 1 | 4 + 6 = 10 | C, F | C (`when "C"`) |
| 2 | 1 + 2 = 3 | F, F | F (`else`) |
