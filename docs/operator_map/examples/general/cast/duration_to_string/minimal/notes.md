# `cast(Me_1, string)` on a duration measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows duration to string with an optional mask; this example gives none, so the default behaviour applies.

> From **duration**: the `duration` value is printed using the ISO-8601 representation for durations.

The inputs use the one-character codes of the durations (the mask section: "A Year", "M Month", "D Day"), which vtlengine reads. The ISO-8601 durations of one year, one month and one day are `P1Y`, `P1M` and `P1D`.

| Me_1 | Me_2 | why |
|---|---|---|
| `A` | `P1Y` | one year |
| `M` | `P1M` | one month |
| `D` | `P1D` | one day |
