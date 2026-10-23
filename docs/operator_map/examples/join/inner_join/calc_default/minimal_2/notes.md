# inner_join: calc_default (minimal_2)

Authored. The `calc` clause overwrites the existing `At_1`, an Attribute of `DS_1`, with the role left out.

Inputs: `DS_1` has the identifiers `Id_1`, `Id_2`, the measure `Me_1` and the **Attribute**
`At_1`; `DS_2` has the same identifiers and the measure `Me_2`. `(2,B)` is in `DS_1` only.

| Id_1 | Id_2 | DS_1 Me_1 | DS_1 At_1 | DS_2 Me_2 |
|---|---|---|---|---|
| 1 | A | 10 | a | 1 |
| 1 | B | 20 | b | 2 |
| 2 | A | 30 | c | 3 |
| 2 | B | 40 | d | |

The inner join keeps `(1,A)`, `(1,B)`, `(2,A)`. `At_1` comes from one operand, so VDS₁ keeps it
("The remaining components, identifier or not, coming from exactly one input data set, which
appear once and maintain their original name and role").

The join page: "**calc** calculates new components by independent sub-expressions evaluated at component level. The role of the new components can be specified, so the original role can be changed if the name is kept the same; if the role is not specified, it will be **measure**", and "If a generated component already exists in any of the input data sets, all input components homonymous to the former are removed, and only the generated component is kept". So `At_1` is a Measure in the result. Outside a join, the Calculation of a Component page gives the opposite default: "if the component exists in the operand Data Set then it maintains its role". Each value of `At_1` gets "x" appended.

| Id_1 | Id_2 | Me_1 | Me_2 | At_1 |
|---|---|---|---|---|
| 1 | A | 10 | 1 | ax |
| 1 | B | 20 | 2 | bx |
| 2 | A | 30 | 3 | cx |

Manual page: [Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html).
