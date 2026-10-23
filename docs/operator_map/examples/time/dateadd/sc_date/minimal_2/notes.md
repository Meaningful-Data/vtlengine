Manual: [Add time unit to date](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Add%20time%20unit%20to%20date/index.html): "adding years to a given date returns the date plus
integer years; for years the “Y” is used", and `dateadd (2020-02-10, 1, “Y”)` "gives
2021-02-10".

**Adaptation.** The page writes the year period indicator `“Y”`; this example writes it `"A"`.
VTL prescribes no representation for durations. The chapter introduction lists the period
indicators used in its examples (`D`, `W`, `M`, `Q`, `S`, and `A` for a year) and says "these
representation are not prescribed by VTL and are not part of the VTL standard, each VTL system
can personalize the representation of time, date, time_period and duration as desired".
vtlengine's data types documentation gives the same codes for Duration (`A`, `S`, `Q`, `M`, `W`,
`D`). The manual's default conversion from string to duration (Type conversion) also reads a
string that does not start with `P` as one of `A`, `S`, `Q`, `M`, `W` or `D`. The page's
original call is:

    dateadd (2020-02-10, 1, “Y”)

The expected result is the page's: adding one year to 2020-02-10 gives 2021-02-10.
