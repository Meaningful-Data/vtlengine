Authored. Same as `minimal`, with the scalar in thenOperand and the Data Set in elseOperand.

Derivation (Case `content.rst`, Behaviour at Data Set level l. 91-95):

- DS_c, as in `minimal`: Id_1 = 1 -> false, 2 -> true, 3 -> true.
- A true data point returns the scalar thenOperand for all the measures (l. 93-95): (2, 0, 0) and
  (3, 0, 0).
- A data point for which no `when` condition is true returns the data point of elseOperand DS_1:
  (1, 5, 50).
