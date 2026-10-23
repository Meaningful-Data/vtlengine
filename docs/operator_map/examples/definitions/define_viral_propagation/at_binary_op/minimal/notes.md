# define viral propagation: propagation at a binary operator

The combination point is a binary operator between two datasets, here the subtraction,
whose Measure result depends on the order of the operands while the propagation does not (User
Manual, the commutative law of the propagation algorithm, "it holds for every operator, also
for non-commutative operations like difference").

`DS_1 - DS_2` pairs the data points with the same `Id_1`: `Me_1` is 10 - 1 = 9 and 20 - 2 = 18,
and `At_1` is "calculated by applying the proper Attribute propagation algorithm on the values of
the Attributes of DP_1 and DP_2" ([Typical behaviours](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html)).

Enumerated rules ([define viral propagation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html)): "A binary clause (e.g., when "A" and "B"
then "C") matches if the input pair consists of exactly those two values. A unary clause (e.g.,
when "A" then "A") matches if the specified value is present in the input pair (regardless of
what the other value is). The first matching clause determines the result." The clauses "are
evaluated in order, but binary clauses (specifying a pair of values) always take precedence over
unary clauses", and "an else clause can provide the default result for any combination that is
not explicitly listed; when it is absent, missing combinations yield NULL".

| Id_1 | DS_1 At_1 | DS_2 At_1 | matching clause | DS_r At_1 |
|---|---|---|---|---|
| 1 | C | F | `when "C"` | C |
| 2 | F | F | none: `else` | F |
