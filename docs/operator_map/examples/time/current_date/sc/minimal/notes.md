Manual: [Actual time](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Actual%20time/index.html), Example 1.

"The operator **current_date** returns the current time as a *date* type." The value changes
with the day of the run, so the example is checked by property (`check="current_date"`): the
result must be a *date* and equal the date of the run. The `null` value of
`expected/scalars.json` is a placeholder that the check does not read.
