Authored. Dataset-level `case` with one `when` whose condition is a named Data Set (`DS_c`), and
an elseOperand (DS_3) that lacks one identifier of the condition (Id_1 = 1). The condition is
named so that the example does not depend on how an engine handles the inline condition of
`minimal`, and shows only how the branches are joined.

Derivation (Case `content.rst` l. 97-99: "its true Data Points are inner joined with thenOperand
and its false Data Points are inner joined with elseOperand, finally the union is made of these
two partial results"; typical behaviour page: "Data Points that are not coupled are left out"):

- DS_c = `DS_1 > 10`: Id_1 = 1 -> false (5), 2 -> true (15), 3 -> true (25), 4 -> false (8).
- True data points (2, 3) with thenOperand DS_2: (2, 200) and (3, 300).
- False data points (1, 4) with elseOperand DS_3, which has Id_1 = 2, 3 and 4 but not 1: (4, 44).
  Id_1 = 1 has no partner in DS_3, so it is left out.
- The union is (2, 200), (3, 300), (4, 44). DS_3's values for Id_1 = 2 and 3 are not used, as
  those data points are true.
