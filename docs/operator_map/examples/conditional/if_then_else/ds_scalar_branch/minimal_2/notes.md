Authored. Same as `minimal`, with the scalar in thenOperand and the Data Set in elseOperand
(l. 78-79: "at least one of thenOperand and elseOperand is a Data Set (the other one can be
scalar)").

Derivation (if-then-else `content.rst`, Behaviour at Data Set level l. 95-99):

- Condition Data Set, as in `minimal`: Id_1 = 1 -> false, 2 -> true, 3 -> true.
- A true data point returns the scalar thenOperand for all the measures: (2, 0, 0) and (3, 0, 0).
- A false data point returns the data point of elseOperand DS_1: (1, 5, 50).
