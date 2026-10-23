# cross_join: rename (manual example 1)

Manual: [Cross Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html),
example 1:
`DS_r := cross_join (DS_1 as d1, DS_2 as d2 rename d1#Id_1 to Id_11, d1#Id_2 to Id_12, d2#Id_1 to Id_21, d2#Id_2 to Id_22, d1#Me_2 to Me_12 );`

Mapped to `rename`, its only clause. The renames name the prefixed components through the
aliases.

The expected output is the manual's, imported as it is. Re-derived from the page:

- The cartesian product has 3 x 3 = 9 data points ("The final result has a size equal to the
  product of the sizes of each input data set").
- `Id_1`, `Id_2` and `Me_2` are in both operands, so VDS₁ has `d1#Id_1`, `d1#Id_2`, `d2#Id_1`,
  `d2#Id_2`, `d1#Me_2`, `d2#Me_2`, plus `Me_1` (from `DS_1` only) and `Me_1A` (from `DS_2` only).
- The rename gives the four identifiers and `d1#Me_2` new names; the prefix of `d2#Me_2` is
  stripped at the end, so it is `Me_2`. Without the renames the stripping would give two `Id_1`,
  two `Id_2` and two `Me_2`, an error.

Every pair of a data point of `DS_1` (`Id_11`, `Id_12`, `Me_1`, `Me_12`) with one of `DS_2`
(`Id_21`, `Id_22`, `Me_1A`, `Me_2`): 9 rows, the manual's `ex_1.csv`.
