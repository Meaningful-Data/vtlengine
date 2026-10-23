# define viral propagation: propagation at an aggregation

The combination point is an aggregation. In [the aggregate invocation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html): "For invocation
at Data Set level, the Attribute propagation rule is applied", and `group by Id_1` keeps only
`Id_1`. The User Manual: "Whenever ... two or more Data Points ... are combined as for their
Measures to give one output Data Point, the default propagation algorithm associated to the viral
Attribute is applied ... This happens for example for the unary Operators which aggregate Data
Points".

Enumerated rules ([define viral propagation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html)): "A binary clause (e.g., when "A" and "B"
then "C") matches if the input pair consists of exactly those two values. A unary clause (e.g.,
when "A" then "A") matches if the specified value is present in the input pair (regardless of
what the other value is). The first matching clause determines the result." The clauses "are
evaluated in order, but binary clauses (specifying a pair of values) always take precedence over
unary clauses", and "an else clause can provide the default result for any combination that is
not explicitly listed; when it is absent, missing combinations yield NULL".

| Id_1 | data points | `sum(Me_1)` | At_1 values | DS_r At_1 |
|---|---|---|---|---|
| 1 | A, B | 10 + 5 = 15 | C, F | C (`when "C"`) |
| 2 | A, B | 3 + 4 = 7 | F, F | F (`else`) |
