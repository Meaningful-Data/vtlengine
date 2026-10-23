# left_join: keep (manual example 1)

Manual: [Left Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html),
example 1: `DS_r := left_join ( DS_1 as d1, DS_2 as d2 keep Me_1, d2#Me_2, Me_1A );`

Mapped to `keep`, its only clause; the aliases are there so that `d2#Me_2` can name one of the
two `Me_2`.

The expected output is the manual's, imported as it is. Re-derived from the page:

- `DS_2` has exactly the identifiers of `DS_1`, so "the `using` clause is optional" and the join
  keys are `Id_1`, `Id_2`. The left-outer join keeps the three keys of `DS_1`: `(1,A)`, `(1,B)`,
  `(2,A)`. `(2,A)` is not in `DS_2`, so "Measures coming from that data set take the null value
  of their respective domains": `d2#Me_2` and `Me_1A` are null there.
- `keep Me_1, d2#Me_2, Me_1A` removes `d1#Me_2`; the prefix of `d2#Me_2` is stripped.

| Id_1 | Id_2 | Me_1 | Me_2 | Me_1A |
|---|---|---|---|---|
| 1 | A | A | Q | B |
| 1 | B | C | T | S |
| 2 | A | E | null | null |

This is the manual's `ex_1.csv`.
