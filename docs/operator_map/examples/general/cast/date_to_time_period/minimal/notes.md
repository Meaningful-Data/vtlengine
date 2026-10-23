# `cast(Me_1, time_period)` on a date measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows date to time_period as an explicit cast without mask.

> From **date** to **time_period**: A `time_period` value with a daily duration is returned, starting and ending at the specified `date` value.

A daily period is written `YYYYD<day of the year>`.

| Me_1 | Me_2 | why |
|---|---|---|
| `2020-01-15` | `2020D15` | day 15 of 2020 |
| `2020-03-01` | `2020D61` | day 61 of 2020 (31 days of January, 29 of February) |
| `2021-12-31` | `2021D365` | day 365 of 2021, not a leap year |
