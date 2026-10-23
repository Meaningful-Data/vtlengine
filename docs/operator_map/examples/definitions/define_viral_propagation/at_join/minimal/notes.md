# define viral propagation: propagation at a join

The combination point is a join. In [inner_join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html): "The Attributes explicitly calculated
through the calc or aggr clauses are maintained unchanged. Other viral attributes, present in
exactly one input data set, are also kept unchanged. For all the other viral attributes, which
are present in multiple data sets, the Attribute propagation rule is applied". `At_1` is viral
in both `DS_1` and `DS_2`, so its two values are combined into one `At_1`, which stays viral.
The Measures have different names (`Me_1`, `Me_2`), so the join keeps both.

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
