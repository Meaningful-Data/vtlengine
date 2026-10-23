Authored from the page's scalar semantics (Case `content.rst` l. 31-32): "The case operator
returns the first thenOperand whose corresponding condition evaluates to true". Here both
conditions are true (15 > 10 and 15 > 1), as in the usual descending-threshold form of a `case`;
the first one wins, so the result is 10, not the 1 of the second `when`. In the other examples of
this variant at most one condition is true.

No recorded team decision applies, so the expected output follows the manual's "first". The
engines implement the opposite for overlapping conditions: "last-match-wins", as the comment of
`visit_Case` in the DuckDB transpiler says.
