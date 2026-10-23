# check_hierarchy: manual example 1

Manual page: [Check hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html), example 1.

`check_hierarchy ( DS_1, HR_1 rule Id_2 partial_null all )`: the ruleset is defined on the
value domain `VD_1`, so `rule Id_2` is mandatory. Mapped to the `rule` variant (the only manual
example that uses the parameter); mode `partial_null`, input `dataset` (default), output `all`.

## Value domain copied from the page text

The page's `intro.rst` says: "the Identifier Id_2 is defined on a valuedomain VD_1 consisting of
the 26 uppercase letters of the English language". `value_domains/VD_1.json` holds those 26
letters (type String). The ruleset `HR_1` is already in the manual's `ex_1.vtl`.

## Erratum

The manual's expected output has, for rule `R030`:

    2010,C,R030,true,0,XX,

`R030 : C = P + Q errorcode "XX" errorlevel 5` gives `0 = 7 + (-7)`, which is **true**. The page's
Behaviour says: "the Measure *errorcode*, which contains the *errorcode* value specified in the
Rule; if the Measure bool_var is **true** or **null** then errorcode is **null**". So `errorcode`
must be null, as `errorlevel` already is in the manual's own row. The corrected row is:

    2010,C,R030,true,0,,

## Check of the other rows (partial_null, input dataset)

`partial_null`: a rule gives a data point when at least one of its code items is found (whichever
its value); a missing item counts as null. Found items: A 5, B 11, C 0, G 19, H null, I 14, M 2,
N 5, O 4, P 7, Q -7, S 3, T 9, U null, V 6.

| rule | relation | evaluation | result |
|---|---|---|---|
| R010 | A = J + K + L | J, K, L missing → null | null |
| R020 | B = M + N + O | 11 = 2 + 5 + 4 | true, imbalance 0 |
| R030 | C = P + Q | 0 = 7 - 7 | true, imbalance 0 |
| R040 | D = R + S | D, R missing (S found) → null | null |
| R050 | E = T + U + V | E missing, U null → null | null |
| R060 | F = Y + W + Z | nothing found | no data point |
| R070 | G = B + C | 19 = 11 + 0 | false, imbalance 8 |
| R080 | H = D + E | H found (null) → null | null |
| R090 | I = D + G | D missing → null | null |
| R100 | M >= N | 2 >= 5 | false, imbalance -3, errorlevel 5 |
| R110 | M <= G | 2 <= 19 | true, imbalance -17 |

Rule R080 shows that the manual counts a found data point with a null measure as "found"
(the Behaviour's reading, not the mode table's "if at least one of the involved Data Points is
not null"). Every row other than R030 matches the manual.
