# `cast(Me_1, date)` on a time_period measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows time_period to date as an explicit cast without mask.

> From **time_period** to **date**: A `date` value is returned if the input `time_period` has a duration of a single day, representing that day; otherwise an error is raised.

Both inputs are daily periods, so none raises the error.

| Me_1 | Me_2 | why |
|---|---|---|
| `2020D15` | `2020-01-15` | day 15 of 2020 |
| `2020D61` | `2020-03-01` | day 61 of 2020, a leap year |
