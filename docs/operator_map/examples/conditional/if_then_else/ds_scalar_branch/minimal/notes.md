Authored. Dataset-level `if` whose elseOperand is a scalar and whose thenOperand is a Data Set.

Derivation (if-then-else page, Behaviour and Additional Constraints):

- `DS_1#Me_1` keeps the identifier Id_1 and the measure Me_1; `> 10` turns it into the condition
  Data Set (Id_1, one boolean measure): Id_1 = 1 -> false (5 is not above 10), 2 -> true (15),
  3 -> true (25).
- A true data point returns the data point of thenOperand DS_1: (2, 15, 150) and (3, 25, 250).
- A false data point returns the scalar elseOperand, which "feeds the values of all the Measures
  of the result Data Point": (1, 0, 0), so both Me_1 and Me_2 are 0.
- DS_1 has two measures on purpose, to show that the scalar feeds all of them. They are Integer,
  like the scalar 0 ("the Measures of the operand Data Set must be all of the same scalar type
  as the scalar operand").
