# `cast(Me_1, time_period)` on a time measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows time to time_period as an explicit cast without mask.

> From **time** to **time_period**: A `time_period` value is returned if the time value has a regular duration (see above for its meaning) otherwise an error is raised.

The page gives no definition of a regular duration above that sentence. The inputs are a whole year, a whole quarter and a whole month, the durations that the page accepts for a `time_period` parsed from a `/` interval ("if its duration is an entire year, a half of a year, a quarter of a year or a month"). The comparison takes `2020A` and `2020` as the same period.

| Me_1 | Me_2 | why |
|---|---|---|
| `2020-01-01/2020-12-31` | `2020A` | the whole year 2020 |
| `2020-04-01/2020-06-30` | `2020Q2` | the second quarter of 2020 |
| `2020-07-01/2020-07-31` | `2020M07` | July 2020 |
