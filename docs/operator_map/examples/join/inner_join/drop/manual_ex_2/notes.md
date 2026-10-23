# inner_join: drop (manual example 2)

Manual: [Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html),
example 2:
`DS_r := inner_join (DS_1 as d1, DS_2 as d2 filter Me_1 = "A" calc Me_4 := Me_1 || Me_1A drop d1#Me_2);`

Mapped to `drop`. The example also has a `filter` and a `calc`, but it is the only manual example
of the page with `drop`; the `filter` and `calc` forms have authored examples of their own (and
`calc` without a role has manual example 3).

The expected output is the manual's, imported as it is. Re-derived from the page:

- Join keys `Id_1`, `Id_2`; the inner join keeps `(1,A)` and `(1,B)`. VDS₁ has `Me_1`, `Me_1A`,
  `d1#Me_2` and `d2#Me_2`.
- `filter Me_1 = "A"` keeps `(1,A)` only.
- `calc Me_4 := Me_1 || Me_1A`: no role is given, so "it will be **measure**"; `"A" || "B"` is
  `AB`.
- `drop d1#Me_2` removes `DS_1`'s `Me_2`; the prefix of `d2#Me_2` is stripped, so `Me_2` is `Q`.

| Id_1 | Id_2 | Me_1 | Me_1A | Me_2 | Me_4 |
|---|---|---|---|---|---|
| 1 | A | A | B | Q | AB |

This is the manual's `ex_2.csv`.
