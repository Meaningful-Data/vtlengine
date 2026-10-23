# define viral propagation: `aggregate avg` across two operators

**Team reading (📌 #1175).** The manual gives two readings for `avg` over more than two values:
the rule "is applied recursively to the values pair by pair. The final result is independent of
the order in which the pairs are evaluated", and the aggregate "must collapse the set of operand
values to a single result"; for `avg` they differ. Issue #1175 was closed on 2026-10-22: "No
solution will be made on this as the standard will be clarified that the avg relies on the
execution order. See https://github.com/sdmx-twg/vtl/issues/744". The clarification proposed
there reads: "Within one operator, avg is applied once to the whole set of operand values, so the
result does not depend on the order of the operands. Across operators, the propagation follows
the order of execution (precedence, associativity and parentheses), so the result depends on
how the operations are grouped."

`(DS_1 + DS_2) + DS_3` has two operators, so `At_1` follows their order of execution: first the
pair of `DS_1` and `DS_2`, then that result with `DS_3`. The User Manual (Properties of the
Attribute propagation algorithm) says the same: "In an expression, the default propagation of
the Attributes is performed always in the same order of execution of the Operators of the
expression, which is determined by their precedence and associativity rules, as already
explained in the relevant section." The Measures are summed (Typical
behaviours, operators on two Data Sets).

| Id_1 | DS_1 At_1 | DS_2 At_1 | DS_3 At_1 | `DS_1 + DS_2` | `+ DS_3` | whole set (not this reading) |
|---|---|---|---|---|---|---|
| 1 | 1.0 | 2.0 | 6.0 | avg(1, 2) = 1.5 | avg(1.5, 6) = 3.75 | 3 |
| 2 | 6.0 | 2.0 | 1.0 | avg(6, 2) = 4 | avg(4, 1) = 2.5 | 3 |

`Me_1`: 10 + 1 + 100 = 111 and 20 + 2 + 200 = 222.
