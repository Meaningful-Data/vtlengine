# check_hierarchy: errorcode

Manual page: [Check hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html).

`DS_1` and the rules are those of the `<mode>_<input>_<output>` grid (A = 10, B = 3, C = null,
D = 6, E = 0, F = 5; X, Y, Z missing), with an errorcode on every rule: rule R<n> gives
errorcode "E<n>". Mode `partial_zero` with output `all` shows
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

"the Measure *errorcode*, which contains the *errorcode* value specified in the Rule; if the
Measure bool_var is **true** or **null** then errorcode is **null**": R3, R4 and R6 (false) carry
theirs; R1, R2 (null) and R7 (true) have a null errorcode.

`Me_1` is Number, so `imbalance` is Number. Subtraction: "If one of the operands is of type
*number*, then the other operand is implicitly cast to *number* and therefore the result has
type *number*".

`errorlevel` (null here) may be Integer or Number: "VTL assumes that a Value Domain
errorlevel_vd of the error levels exists in the Information Model" (define hierarchical
ruleset), so its type is the Information Model's choice; the `errorlevel` variant tests it.
