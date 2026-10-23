# inner_join: apply (manual example 4)

Manual: [Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html),
example 4: `DS_r := inner_join (DS_1 as d1, DS_3 as d2 apply d1 || d2);`

Mapped to `apply`, its only clause. It uses the aliases in the expression; the authored example
of the variant uses the data set names, which the page also allows ("The expression *applyExpr*
can use as input the names or aliases of the operand data sets").

The expected output is the manual's, imported as it is. Re-derived from the page:

- Join keys `Id_1`, `Id_2`; the inner join keeps `(1,A)` and `(1,B)`.
- Both operands have the measures `Me_1` and `Me_2`: "**apply** combines the homonym Measures in
  the source operands", here `Me_1 := d1#Me_1 || d2#Me_1` and `Me_2 := d1#Me_2 || d2#Me_2`.

  At `(1,A)` that is `"A" || "B"` and `"B" || "Q"`; at `(1,B)`, `"C" || "S"` and `"D" || "T"`.

| Id_1 | Id_2 | Me_1 | Me_2 |
|---|---|---|---|
| 1 | A | AB | BQ |
| 1 | B | CS | DT |

This is the manual's `ex_4.csv`.
