# `cast(Me_1, date)` on a time measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows time to date as an explicit cast without mask.

> From **time** to **date**: A `date` value is returned if the `time` value start and ends at the same date, otherwise an error is raised.

Every input starts and ends on the same day, so none raises the error.

| Me_1 | Me_2 | why |
|---|---|---|
| `2020-01-15/2020-01-15` | `2020-01-15` | starts and ends on 2020-01-15 |
| `2021-12-31/2021-12-31` | `2021-12-31` | starts and ends on 2021-12-31 |
