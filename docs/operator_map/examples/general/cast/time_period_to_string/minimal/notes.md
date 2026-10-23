# `cast(Me_1, string)` on a time_period measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows time_period to string with an optional mask; this example gives none, so the default behaviour applies.

> From **time_period**: the `time_period` value is printed using the ISO-8601 representation for generic time intervals, up to a precision of days, in the form start/end.

So the default string of a period is its interval, not a period code such as `2020Q1`.

| Me_1 | Me_2 | why |
|---|---|---|
| `2020Q1` | `2020-01-01/2020-03-31` | the first quarter of 2020 |
| `2021M02` | `2021-02-01/2021-02-28` | February 2021 |

## Triage (both engines give `2020Q1` and `2021M2`)

Re-derived from the rule above: without a mask the manual prints a `time_period` as its
interval, "in the form start/end", so the expected file stays. The engines print the period
code instead.
