# check_hierarchy: errorlevel

Manual page: [Check hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html).

`DS_1` and the rules are those of the `<mode>_<input>_<output>` grid (A = 10, B = 3, C = null,
D = 6, E = 0, F = 5; X, Y, Z missing), with an errorlevel on every rule: rule R<n> gives
errorlevel <n>. Mode `partial_zero` with output `all` shows
three false results, a true one and two null ones side by side, and no rule without any item
(the question that separates `always_*` from `partial_*` stays in the grid).

| rule | relation | result (partial_zero, dataset) |
|---|---|---|
| R1 | `A = B + C` | C null: null |
| R2 | `C = D + E` | C null: null |
| R3 | `B >= D` | `3 >= 6`: **false**, imbalance -3 |
| R4 | `E > X` | X missing, counts as 0: `0 > 0` **false**, imbalance 0 |
| R5 | `Y < Z` | nothing found: no data point |
| R6 | `D <= X` | X counts as 0: `6 <= 0` **false**, imbalance 6 |
| R7 | `F > B` | `5 > 3`: true, imbalance 2 |

"the Measure *errorlevel*, which contains the *errorlevel* value specified in the Rule; if the
Measure bool_var is **true** or **null** then errorlevel is **null**": R3, R4 and R6 (false) carry
theirs; R1, R2 (null) and R7 (true) have a null errorlevel.

`errorlevel` is Integer, the typing of the manual's example structures (check_datapoint
examples 1 and 2, check_hierarchy example 1). The manual leaves the type to the Information
Model ("VTL assumes that a Value Domain errorlevel_vd of error levels exists in the Information
Model", define datapoint ruleset; define hierarchical ruleset says the same), so this variant
keeps the manual's example typing strict on purpose: a difference here is for the maintainer to
decide, and the other examples accept Integer or Number.

`Me_1` is Integer here, as in the manual's example 1, so `imbalance`
is Integer (Subtraction: "If the type of both operands is *integer* then the result has type
*integer*"): this example also tests that typing.
