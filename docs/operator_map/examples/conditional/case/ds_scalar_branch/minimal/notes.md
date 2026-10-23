Authored. Dataset-level `case` whose thenOperand is a Data Set and whose elseOperand is a scalar
("the same rules apply as for the if-then-else operator", l. 79). The condition is a named Data
Set, so that the example is about the scalar branch only (`ds/minimal` has the inline condition).

Derivation (Case `content.rst`, Behaviour at Data Set level l. 91-95; if-then-else l. 83-84 for the
types):

- DS_c = `DS_1#Me_1 > 10`: Id_1 = 1 -> false (5), 2 -> true (15), 3 -> true (25).
- A true data point returns the data point of thenOperand DS_1: (2, 15, 150) and (3, 25, 250).
- A data point for which no `when` condition is true returns the scalar elseOperand, which "feeds
  the values of all the Measures of the result Data Point" (l. 93-95): (1, 0, 0).
- DS_1 has two measures on purpose, to show that the scalar feeds all of them. They are Integer,
  like the scalar 0.
