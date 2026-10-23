Authored. Dataset-level `case` with one `when`. The condition is written inline (`DS_1 > 10`),
like the condition of the if-then-else manual example. "The same rules apply as for the
if-then-else operator" (l. 79).

Derivation (Case `content.rst`, Behaviour at Data Set level l. 91-99):

- The condition Data Set `DS_1 > 10` is (Id_1, one boolean measure): Id_1 = 1 -> false (5),
  2 -> true (15), 3 -> true (25).
- Its true data points are inner joined with thenOperand DS_2 (l. 97-98): (2, 200) and (3, 300).
- Its false data points are inner joined with elseOperand DS_3 (l. 98): (1, 1).
- All three identifiers are in both branch Data Sets, so no data point is left out.
