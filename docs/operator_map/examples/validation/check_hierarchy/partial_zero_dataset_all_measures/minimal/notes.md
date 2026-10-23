# check_hierarchy: partial_zero dataset all_measures

Manual page: [Check hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html)
(Behaviour and the mode table `modeTable.csv`).

The 36 `<mode>_<input>_<output>` examples share `DS_1` and `HR_1`. `DS_1` has one group
(`Id_1` = 1) with A = 10, B = 3, C = null, D = 6, E = 0, F = 5; X, Y and Z are missing. The rules
use every relation: `=` (R1, R2), `>=` (R3), `>` (R4, R7), `<` (R5) and `<=` (R6). `HR_1` gives no
errorcode and no errorlevel, so both are null on every data point (the `errorcode` and
`errorlevel` variants check them).

Mode `partial_zero`: a data point is produced when at least one item of the rule is found; a
missing item counts as 0.
A null data point stays null in every mode (mode table, "Null Data Points are considered:
null"). A comparison or an imbalance with a null item is null.

Input `dataset`: every item is taken from `DS_1` only, so `C` stays null.

| rule | relation | result |
|---|---|---|
| R1 | `A = B + C` | null (Me_1 10) |
| R2 | `C = D + E` | null (Me_1 null) |
| R3 | `B >= D` | false, imbalance -3 (Me_1 3) |
| R4 | `E > X` | false, imbalance 0 (Me_1 0) |
| R5 | `Y < Z` | no data point |
| R6 | `D <= X` | false, imbalance 6 (Me_1 6) |
| R7 | `F > B` | true, imbalance 2 (Me_1 5) |

Output `all_measures`: every data point produced, with `bool_var` and `Me_1`.
`Id_2` is the left item of the rule; imbalance is left minus right.
This example keeps R1, R2, R3, R4, R6, R7.

Types:
- `Me_1` is Number, so `imbalance` is Number. Subtraction: "If one of the operands is of type
  *number*, then the other operand is implicitly cast to *number* and therefore the result has
  type *number*".
- `errorlevel` is Integer even though no rule gives one: the manual's check_datapoint examples 1
  and 2 have no errorlevel in their ruleset and still type it Integer, and `check` says "or
  *integer* if the errorcode_vd Value Domain is not found".

`DS_1` has six data points, more than the 2–4 of the other authored inputs. One dataset must
hold:
- a null item that an `=` rule defines (C, for `dataset_priority`);
- a zero item (E);
- present non-zero items for a false and a true fully present rule (B, D, F);
- a left item of its own for every rule, so that the grid does not depend on how repeated left
  items are treated.

Readings taken where the manual is not explicit:
- the computed output of a rule (input `dataset_priority`) is the value it computes for its
  left item from its right items, as in `hierarchy`;
- a missing left item that a zero mode counts as 0 gives `Me_1` = 0 (R5 in `always_zero`).
