Authored. The same as `minimal`, with the two conditions written as named Data Sets (`DS_c1`,
`DS_c2`), so that the example does not depend on how an engine handles an inline condition and
shows only the order in which the conditions are tried.

Derivation (Case `content.rst`, Behaviour at Data Set level l. 91-92), as in `minimal`:

- Id_1 = 1, DS_1 = 5: `DS_c1` false, `DS_c2` false -> elseOperand DS_1: (1, 5).
- Id_1 = 2, DS_1 = 15: `DS_c1` false, `DS_c2` true -> DS_3: (2, 2).
- Id_1 = 3, DS_1 = 25: `DS_c1` true -> DS_2: (3, 300). `DS_c2` is true as well, but the first true
  condition wins, so the result is not DS_3's 3.

No recorded team decision applies, so the expected output follows the manual's "first". The
engines implement the opposite for overlapping conditions: "last-match-wins", as the comment of
`visit_Case` in the DuckDB transpiler says.
