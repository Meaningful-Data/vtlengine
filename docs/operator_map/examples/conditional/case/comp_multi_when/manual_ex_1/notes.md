Manual example `ex_1` of the Case page, imported as it is. A component-level `case` inside `calc`
with three `when` clauses that return scalars: variant `comp_multi_when`.

Re-derivation (Case page: the result is the thenOperand of the first `when` whose condition is
true, and elseOperand when none is true):

- Me_1 = 0.12: `Me_1 <= 1` is true -> 0.
- Me_1 = 3.5: `Me_1 > 1 and Me_1 <= 10` is true -> 1.
- Me_1 = 10.7: `Me_1 > 10` is true -> 10.
- Me_1 = null: no comparison with null is true -> else, 100.

The result is the manual's. The three conditions are mutually exclusive, so this example cannot
tell the first true `when` from the last one: the authored `minimal` example does.
