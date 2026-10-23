Manual: [Number days to duration](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Number%20days%20to%20duration/index.html), `ex_1`.

**Erratum.** The manual's expected output repeats `Id_1 = G` on its second data point, where the
input DS_1 has `H` (the calc keeps the identifiers of DS_1, and two data points cannot share the
key `G`):

```
Id_1,Me_1,Me_2
G,240,P0Y240D
G,724,P1Y359D
T,1056,P2Y326D
```

The expected output has `H,724,P1Y359D`. The durations are the manual's and follow its rule
("according to ISO 8601 Y = 365D", mask `\PY\YDDD\D`): 240 = 0 × 365 + 240, 724 = 1 × 365 + 359,
1056 = 2 × 365 + 326.
