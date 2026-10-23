# full_join: keep (manual example 1)

Manual: [Full Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html),
example 1: `DS_r := full_join ( DS_1 as d1, DS_2 as d2 keep Me_1, d2#Me_2, Me_1A );`

Mapped to `keep`, its only clause; the aliases are there so that `d2#Me_2` can name one of the
two `Me_2`.

The expected output is the manual's, imported as it is. Re-derived from the page:

- The operands have exactly the same identifiers, so "the using clause is optional" and the join
  keys are `Id_1`, `Id_2`. The full outer join keeps `(1,A)` and `(1,B)` (in both), `(2,A)`
  (`DS_1` only: the measures of `DS_2` are null) and `(3,A)` (`DS_2` only: "an additional data
  point is added ... for each data point in the current join operand that wasn't matched
  before", with the measures of `DS_1` null).
- `keep Me_1, d2#Me_2, Me_1A` removes `d1#Me_2`; the prefix of `d2#Me_2` is stripped.

| Id_1 | Id_2 | Me_1 | Me_2 | Me_1A |
|---|---|---|---|---|
| 1 | A | A | Q | B |
| 1 | B | C | T | S |
| 2 | A | E | null | null |
| 3 | A | null | M | Z |

This is the manual's `ex_1.csv`.
