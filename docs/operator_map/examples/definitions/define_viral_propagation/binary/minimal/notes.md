# define viral propagation: a binary clause

The binary clause `when "C" and "M"` (`enumerationCondition ::= enumerationTerm [ and
enumerationTerm ]`) is written after the unary `when "M"` on purpose: "binary clauses ... always
take precedence over unary clauses", so the pair (C, M) gives N, like the manual's
`COMPUTED_missing` example ("The binary rule "C" and "M" takes precedence over the unary rule
"M""). The pair matches in either order: the input is "exactly those two values".

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
| 1 | C | M | `when "C" and "M"` (binary first) | N |
| 2 | M | C | `when "C" and "M"` (same pair) | N |
| 3 | M | F | `when "M"` (not the pair C, M) | M |
| 4 | C | C | none: `else` | F |

Like the manual's `COMPUTED_missing`, this rule is not associative over three values (M with
the N of a previous (C, M) gives M), which the manual asks of a rule ("the rule must be defined
such that the result is consistent regardless of grouping"). Here it only combines pairs, so
the results above are determined.
