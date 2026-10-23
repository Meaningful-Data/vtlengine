# Manual example 1: `hierarchy ( DS_1, HR_1 rule Id_2 non_null )`

Manual page: [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html), example 1. Mode `non_null`, with the default input
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

`non_null`: a missing Data Point counts as null, and a rule gives a Data Point only when its value
is not null (zeros are returned). With input `rule`, B, C, D, E and G come from the rules that
compute them.

| Rule | Right side | Me_1 | Returned | At_1 |
|---|---|---|---|---|
| A = J + K + L | all missing | null | no | |
| B = M + N + O | 2 + 5 + 4 | 11 | yes | Dx (Dx, Pz, Pz) |
| C = P + Q | 7 + (-7) | 0 | yes | Pz |
| D = R + S | R missing | null | no | |
| E = T + U + V | U is null | null | no | |
| F = Y + W + Z | all missing | null | no | |
| G = B + C | 11 + 0 | 11 | yes | Dx (Dx, Pz) |
| H = D + E | D, E not computed | null | no | |
| I = D + G | D not computed | null | no | |

The result (B, C, G) is the page's.
