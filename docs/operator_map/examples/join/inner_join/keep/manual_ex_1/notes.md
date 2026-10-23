# inner_join: keep (manual example 1)

Manual: [Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html),
example 1: `DS_r := inner_join (DS_1 as d1, DS_2 as d2 keep Me_1, d2#Me_2, Me_1A );`

Mapped to `keep`: `keep` is the only clause of the example. The aliases are there so that
`d2#Me_2` can name one of the two `Me_2`.

The expected output is the manual's, imported as it is. Re-derived from the page:

- The join keys are `Id_1` and `Id_2`, the identifiers of both operands (natural join). The inner
  join keeps the keys found in both: `(1,A)` and `(1,B)`; `(2,A)` is in `DS_1` only and `(3,A)` in
  `DS_2` only.
- VDS₁ has `Me_1` (from `DS_1` only), `Me_1A` (from `DS_2` only) and, since `Me_2` is in both
  operands, `d1#Me_2` and `d2#Me_2`: "names of each of these components are prefixed with the
  alias of the data set they come from".
- "**keep** will maintain all the identifiers and only the specified non-identifier components":
  `d1#Me_2` goes. Then "all components that originally appeared in multiple input data sets, are
  renamed by stripping their previously determined prefix": `d2#Me_2` becomes `Me_2`.

| Id_1 | Id_2 | Me_1 | Me_2 | Me_1A |
|---|---|---|---|---|
| 1 | A | A | Q | B |
| 1 | B | C | T | S |

This is the manual's `ex_1.csv`.
