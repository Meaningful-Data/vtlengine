# `cast(Me_1, time)` on a date measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows date to time implicitly.

> From **date** to **time**: a `time` value is returned having its start equal to its end, and both equal to the input `date` value.

The conversion is implicit, and the example writes it as an explicit cast: "An **explicit cast without mask** can also be used to make an implicit cast more clear, and in this case its behavior remains the same." A `time` value is written `start/end`, as vtlengine reads it.

| Me_1 | Me_2 | why |
|---|---|---|
| `2020-01-15` | `2020-01-15/2020-01-15` | start and end on the date |
| `2021-12-31` | `2021-12-31/2021-12-31` | start and end on the date |
