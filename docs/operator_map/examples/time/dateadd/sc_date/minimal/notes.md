Manual: [Add time unit to date](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Add%20time%20unit%20to%20date/index.html): "adding months to a given date returns the date plus
integer months", and `dateadd (2020-02-10, 1, “M”)` "gives 2020-03-10". The date is the cast
of a string (VTL has no date literal), and the period indicator a string, as in the page's
`ex_1` (`dateadd(Me_1, 2, "M")`).
