# inner_join: calc_measure (manual example 3)

Manual: [Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html),
example 3:
`DS_r := inner_join ( DS_1  filter Id_2 ="B" calc Me_2 := Me_2 || "_NEW" keep Me_1, Me_2);`

Mapped to `calc_measure`: its `calc` gives no role, and in a join "if the role is not specified,
it will be **measure**", so it is the role-less form of the variant whose authored example writes
`measure` out. It overwrites `Me_2`, which is in the input. The `filter` and `keep` of the example
have authored examples of their own (and `keep` has manual example 1).

The expected output is the manual's, imported as it is. Re-derived from the page:

- The join has one operand, so VDS₁ is `DS_1`.
- `filter Id_2 = "B"` keeps `(1,B)`, where `Me_1` = C and `Me_2` = D.
- `calc Me_2 := Me_2 || "_NEW"`: "If a generated component already exists in any of the input
  data sets, all input components homonymous to the former are removed, and only the generated
  component is kept", so `Me_2` = `D_NEW`.
- `keep Me_1, Me_2` keeps both measures.

| Id_1 | Id_2 | Me_1 | Me_2 |
|---|---|---|---|
| 1 | B | C | D_NEW |

This is the manual's `ex_3.csv`.

**A doubt in the manual, not corrected.** The syntax of the page is
`inner_join ( ds1 { as alias1 }, ds2 { as alias2 } { , dsN { as aliasN } }* ...` and its input
parameters say "ds1, ..., dsN: a minimum of two data set operands", but this example has one
operand. The official VTL 2.2 grammar (`joinClause: joinClauseItem (COMMA joinClauseItem)*?`)
accepts one operand, and the manual prints a result for it, so the example is kept as it is.
