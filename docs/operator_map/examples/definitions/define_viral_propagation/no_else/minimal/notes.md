# define viral propagation: no `else` clause

The rule has no `else`: "If omitted and no enumerated clause matches, NULL is
propagated".

`DS_1 + DS_2` pairs the data points with the same `Id_1` and adds their Measures ([Typical
behaviours](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html), operators on two Data Sets). The viral Attribute `At_1` of each pair is
"calculated by applying the proper Attribute propagation algorithm on the values of the
Attributes of DP_1 and DP_2", and it stays viral in the result ("DS_r maintains the Attributes
declared as viral for the operand Data Sets; these Attributes are considered as viral also in
DS_r").

Enumerated rules ([define viral propagation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html)): "A binary clause (e.g., when "A" and "B"
then "C") matches if the input pair consists of exactly those two values. A unary clause (e.g.,
when "A" then "A") matches if the specified value is present in the input pair (regardless of
what the other value is). The first matching clause determines the result." The clauses "are
evaluated in order, but binary clauses (specifying a pair of values) always take precedence over
unary clauses", and "an else clause can provide the default result for any combination that is
not explicitly listed; when it is absent, missing combinations yield NULL".

| Id_1 | DS_1 At_1 | DS_2 At_1 | matching clause | DS_r At_1 |
|---|---|---|---|---|
| 1 | C | N | `when "C"` | C |
| 2 | N | N | none | null |
| 3 | N | A | none | null |

Row 2, the pair (N, N), combines two equal values into a different one (null). The section calls the algorithm "reflexive", but the User Manual's reflexive law is about one operand ("A( §(D1)) = A(D1)"), and the manual's own examples also give another value for a pair of equal values (`OBS_STATUS_default` gives F for (E, E), `COMPUTED_missing` gives " " for (C, C)), so the example follows the clauses as written.
