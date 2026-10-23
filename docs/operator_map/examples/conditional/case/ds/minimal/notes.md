Authored. Dataset-level `case` with one `when`. The page says the same rules apply as for
if-then-else.

Derivation (Case page, Behaviour at Data Set level):

- The condition Data Set `DS_1 > 10` is (Id_1, one boolean measure): Id_1 = 1 -> false (5),
  2 -> true (15), 3 -> true (25).
- A data point whose `when` condition is true returns the data point of its thenOperand DS_2:
  (2, 200) and (3, 300).
- A data point for which no `when` condition is true returns the data point of elseOperand DS_3:
  (1, 1).
