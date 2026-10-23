# `cast(Me_1, time)` on a time_period measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows time_period to time implicitly.

> From **time_period** to **time**: a `time` value is returned having the stame start and end of the input `time_period` value.

The conversion is implicit, and the example writes it as an explicit cast: "An **explicit cast without mask** can also be used to make an implicit cast more clear, and in this case its behavior remains the same." A `time` value is written `start/end`, as vtlengine reads it.

| Me_1 | Me_2 | why |
|---|---|---|
| `2020Q1` | `2020-01-01/2020-03-31` | the first quarter of 2020 |
| `2021M02` | `2021-02-01/2021-02-28` | February 2021, not a leap year |
