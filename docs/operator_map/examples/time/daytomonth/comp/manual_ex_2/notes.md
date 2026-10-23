Manual: [Number days to duration](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Number%20days%20to%20duration/index.html), `ex_2`.

**Erratum.** As in `ex_1`, the manual's expected output repeats `Id_1 = G` on its second data
point, where the input DS_1 has `H`:

```
Id_1,Me_1,Me_2
G,240,P8M0D
G,724,P24M4D
T,1056,P35M6D
```

The expected output has `H,724,P24M4D`. The durations are the manual's and follow its rule
("according to ISO 8601 M = 30D", mask `\PM\MDD\D`): 240 = 8 × 30 + 0, 724 = 24 × 30 + 4,
1056 = 35 × 30 + 6.
