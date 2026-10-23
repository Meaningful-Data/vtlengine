# Manual example 2: `hierarchy ( DS_1, HR_1 rule Id_2 non_zero )`

Manual page: [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html), example 2. Mode `non_zero`, with the default input
`rule` and the default output `computed`.

## Definitions added to the script

The script of the example is only the `hierarchy` call. The page gives the rest in its text
(`examples/intro.rst`), so it is added to `script.vtl`:

- The hierarchical ruleset `HR_1` is copied verbatim from the page.
- The page describes the propagation of the viral attribute `At_1` in prose: "the alphabetic
  order prevails the null prevails on the alphabetic characters and the Attribute value for
  missing Data Points is assumed as null". It is written as an enumerated
  `define viral propagation` (VTL-DL, `vtl_dl_rulesets/viral_attributes`): the clause
  `when null then null` comes first, so a null prevails; then one unary clause per value of
  `At_1` in the data, in alphabetic order, so of two values the alphabetically first one wins.
  The last part of the sentence is not a clause: it says that a missing Data Point brings a null
  `At_1` to the propagation, which the derivation below applies.
- The value domain `VD_1` of the ruleset's signature is not defined on the page (its code list
  is not given), so no value domain file is added.

The input and the expected output are the page's, unchanged.

## Derivation (checked by hand against the page)

`non_zero`: a missing Data Point counts as 0, and a rule's Data Point is returned when its value
is not 0 (nulls are returned too). With input `rule`, B, C, D, E and G come from the rules that
compute them.

| Rule | Right side | Me_1 | Returned | At_1 |
|---|---|---|---|---|
| A = J + K + L | all missing: 0 + 0 + 0 | 0 | no | |
| B = M + N + O | 2 + 5 + 4 | 11 | yes | Dx (Dx, Pz, Pz) |
| C = P + Q | 7 + (-7) | 0 | no (computed, not returned) | Pz |
| D = R + S | R missing: 0 + 3 | 3 | yes | null (R is missing) |
| E = T + U + V | 9 + null + 6 | null | yes | Bq (Bq, Nj, Ko) |
| F = Y + W + Z | all missing | 0 | no | |
| G = B + C | 11 + 0 | 11 | yes | Dx (Dx, Pz) |
| H = D + E | 3 + null | null | yes | null (D's) |
| I = D + G | 3 + 11 | 14 | yes | null (D's) |

The result is the page's. Its `At_1` of G (Dx) shows that G reads C from the computed output of
the rule `C = P + Q` although `non_zero` does not return it: a missing C would bring a null
`At_1`, which would prevail.
