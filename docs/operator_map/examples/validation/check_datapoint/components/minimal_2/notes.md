# check_datapoint: components (ruleset on value domains)

Manual pages: [Check datapoint](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html) and define datapoint ruleset.

The case where `components` matters. `DPR_2` has the rules of `DPR_1` (the `components/minimal`
example) written on value domains, so its signature names value domains, not components:
"If *dpr* is defined on Value Domains then it is mandatory to specify *listComp*", and
*listComp* is "the list of Components of *op* to be associated (in positional order) to the
conditioning Value Domains defined in *dpr*". `components Id_2, Me_1` binds `VD_2` to `Id_2` and
`VD_M` to `Me_1`, so the rules evaluate exactly as those of `DPR_1`:

| data point | R1 `when VD_2 = "A" then VD_M > 0` | R2 `VD_M < 100` |
|---|---|---|
| 1, A, 10 | 10 > 0: true | 10 < 100: true |
| 2, A, -5 | -5 > 0: **false** | -5 < 100: true |
| 3, B, 200 | antecedent false: true | 200 < 100: **false** |
| 4, A, null | null > 0: null | null < 100: null |

The output is the default, `invalid`: (2, A, R1) and (3, B, R2), with the measures of `DS_1`; no
rule gives an errorcode or errorlevel, so both are null. The expected output is the one of
`components/minimal`.

`value_domains/VD_2.json` is the value domain of `Id_2` (String: A, B). `VD_M` is the numeric
value domain of `Me_1`; it has no file, since a value domain file enumerates its values and the
rules never test membership of `VD_M`.

`errorlevel` (null here) may be Integer or Number: "VTL assumes that a Value Domain
errorlevel_vd of error levels exists in the Information Model" (define datapoint ruleset), so its
type is the Information Model's choice; the `errorlevel` variant tests it.
