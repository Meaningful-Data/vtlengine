# `cast(Me_1, string)` on a time measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows time to string with an optional mask; this example gives none, so the default behaviour applies.

> From **time**: the `time` value is printed using the ISO-8601 representation for generic time intervals, up to a maximum of day precision, in the form start/end.

The dates are written in the ISO-8601 extended form `YYYY-MM-DD`.

| Me_1 | Me_2 | why |
|---|---|---|
| `2020-01-01/2020-06-30` | `2020-01-01/2020-06-30` | start/end |
| `2021-03-15/2021-03-20` | `2021-03-15/2021-03-20` | start/end |
