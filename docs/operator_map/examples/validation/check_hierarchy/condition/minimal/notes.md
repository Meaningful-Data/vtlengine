# check_hierarchy: condition

Manual pages: [Check hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html) and define hierarchical
ruleset.

`HR_2` is defined on variables with the conditioning variable `Id_1`; the invocation names it
with `condition Id_1` (optional for a ruleset on variables, "to show explicitly in the
invocation which are the involved Components"). Mode, input and output are the defaults:
`non_null`, `dataset`, `invalid`. The rule gives no errorcode or errorlevel, so both are null.

The leftCondition `Id_1 = 2` "defines the pre-condition for evaluating the left member Code
Item (i.e., it is evaluated only when the leftCondition is true)":

- `Id_1` = 1: the condition is false, the rule is not evaluated (A = 5, B = 2 would be false);
- `Id_1` = 2: `7 = 4` is false, imbalance 7 - 4 = 3.

Output `invalid` keeps the false data point with `Me_1` = 7 (the left item A). A rule that is not
evaluated gives no invalid data point under either reading of what it returns, so the default
output avoids that question.

`Me_1` is Number, so `imbalance` is Number. Subtraction: "If one of the operands is of type
*number*, then the other operand is implicitly cast to *number* and therefore the result has
type *number*".

`errorlevel` is Integer even where no rule gives one: the manual's check_datapoint examples 1
and 2 have no errorlevel in their ruleset and still type it Integer (and `check` says "or
*integer* if the errorcode_vd Value Domain is not found").
