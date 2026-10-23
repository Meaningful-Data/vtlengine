Manual: [Add time unit to date](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Add%20time%20unit%20to%20date/index.html), the valid syntax `dateadd (2022Q1, 5, “M”)`.

"The last day of the time period is assumed as the starting date": 2022Q1 ends on 2022-03-31.
"adding months to a given date returns the date plus integer months": five months later is
2022-08-31 (August has 31 days). "The scalar version of this operator takes as input one date
or time_period value and returns a date", so the result is a *date*.
