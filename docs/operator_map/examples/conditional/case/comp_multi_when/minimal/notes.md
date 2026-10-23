Authored. Component-level `case` with two `when` clauses whose conditions overlap, the usual
descending-threshold form. The manual example of this variant has mutually exclusive conditions,
so it cannot tell the first true `when` from the last one.

Derivation (Case page: the scalar semantics return "the first thenOperand whose corresponding
condition evaluates to true", "elseOperand if none of the when conditions evaluates to true";
at Component level the operation is applied for each data point, so each value of Me_1 is
treated like the scalar case):

- Id_1 = 1, Me_1 = 0: `0 > 10` false, `0 > 1` false -> elseOperand 0.
- Id_1 = 2, Me_1 = 5: `5 > 10` false, `5 > 1` true -> 1.
- Id_1 = 3, Me_1 = 15: `15 > 10` true -> 10. `15 > 1` is true as well, but the first true
  `when` wins, so the result is 10, not 1.

No recorded team decision applies, so the expected output follows the manual's "first". The
engines implement the opposite for overlapping conditions: "last-match-wins", as the comment of
`visit_Case` in the DuckDB transpiler says.
