Manual: [Duration to number days](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Duration%20to%20number%20days/index.html), `ex_1`.

**Erratum.** The manual's expected data points are those of Number days to duration `ex_1`
(with its repeated `G`), not `yeartoday` of this page's DS_1: the `Me_1` column holds numbers
of days where DS_1 holds durations, and `Me_2` holds durations where the structure says
Integer:

```
Id_1,Me_1,Me_2
G,240,P0Y240D
G,724,P1Y359D
T,1056,P2Y326D
```

The expected output is derived from the page: `yeartoday` "takes as input a duration having
the following mask: `\PY\YDDD\D`" and returns the number of days, "according to ISO 8601
Y = 365D" (the page's Example 2, `yeartoday (P1Y20D)` returns 385, follows the same rule):

- `G`: P2Y230D = 2 × 365 + 230 = 960
- `H`: P1Y23D = 1 × 365 + 23 = 388
- `T`: P3Y152D = 3 × 365 + 152 = 1247

The input durations `P2Y230D`, `P1Y23D` and `P3Y152D` are kept as the manual writes them:
vtlengine's Duration holds only a period indicator (`A`, `S`, `Q`, `M`, `W`, `D`), so it has no
representation of them to adapt to. A failure to load them is an engine finding.
