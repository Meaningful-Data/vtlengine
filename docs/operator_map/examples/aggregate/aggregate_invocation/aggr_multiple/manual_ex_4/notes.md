# Aggregate invocation, ex_4: `DS_1 [ aggr Me_2 := max ( Me_1 ) , Me_3 := min ( Me_1 ) group by Id_1 ]`

Erratum (expected output). The manual's `ex_4.json` / `ex_4.csv`
([Aggregate invocation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html)) describe a data set named
`DS_3` with an extra component `At_1` of role `ViralAttribute`:

| Id_1 | Me_2 | Me_3 | At_1 |
|---|---|---|---|
| 2010 | 23 | 1 | |
| 2011 | 20 | -1 | N |
| 2012 | 40 | 30 | P |

Two corrections:

- The script assigns `DS_r`, so the result is `DS_r` (a copy-paste slip: `DS_3`).
- `DS_1` declares `At_1` with the role `Attribute`, not `ViralAttribute`, and the script defines
  no viral propagation. The User Manual says "If the virality is not defined, the Attribute is
  considered as non-viral" and "the non-viral Attributes are not kept in the result and their
  values are not considered"; the Aggregate invocation adds that at component level "the
  Attributes calculated within the aggr clause are maintained in the result; for all the other
  Attributes that are defined as viral, the Attribute propagation rule is applied". `At_1` is
  neither calculated nor viral, so it is not in the result, as in the other three examples of
  the page, which drop it.

The measures are the manual's: 2010 → max 23, min 1; 2011 → max 20, min -1; 2012 → max 40,
min 30. `max` and `min` keep the Integer type of `Me_1`.
