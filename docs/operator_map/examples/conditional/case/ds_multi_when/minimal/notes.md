Authored. Dataset-level `case` with two `when` clauses whose conditions overlap, written inline,
to check that the first true condition wins.

Derivation (Case `content.rst`, Behaviour at Data Set level l. 91-92: the result is the data
point of the thenOperand of the first condition whose data point with the same identifiers is
true, and the data point of elseOperand when none is true):

- Id_1 = 1, DS_1 = 5: `DS_1 > 20` false, `DS_1 > 10` false -> elseOperand DS_1: (1, 5).
- Id_1 = 2, DS_1 = 15: `DS_1 > 20` false, `DS_1 > 10` true -> DS_3: (2, 2).
- Id_1 = 3, DS_1 = 25: `DS_1 > 20` true -> DS_2: (3, 300). `DS_1 > 10` is true as well, but the
  first true condition wins, so the result is not DS_3's 3.

No recorded team decision applies, so the expected output follows the manual's "first". The
engines implement the opposite for overlapping conditions: "last-match-wins", as the comment of
`visit_Case` in the DuckDB transpiler says.
