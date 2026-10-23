# Analytic invocation, ex_1: `sum ( DS_1 over ( order by Id_1, Id_2, Id_3 data points between 1 preceding and 1 following ) )`

The expected data points are the manual's. The type of `Me_1` allows Integer or Number: the
manual prints Number, and the Sum page says "implementations may preserve the integer type for
the resulting measure or component when all operand values are integer" (`Me_1` of `DS_1` is
Integer).

There is no partition clause, and the order clause names every identifier, so by the rule "If
the clause is omitted, then the Data Set is partitioned by the Identifier Components that are
not specified in the orderClause" the whole data set is one partition. Check, in the order of
the identifiers: -3 + 5 = 2, -3 + 5 + 13 = 15, 5 + 13 + 9 = 27, 13 + 9 + 7 = 29, 9 + 7 + 11 = 27,
7 + 11 - 1 = 17, 11 - 1 + 0 = 10, -1 + 0 + 3 = 2, 0 + 3 - 2 = 1, 3 - 2 = 1.
