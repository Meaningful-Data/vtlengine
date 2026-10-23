Manual: [Add time unit to date](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Add%20time%20unit%20to%20date/index.html), the valid syntax `DS_r := dateadd(DS_1, 1, “M”)`.

The operand is typed `dataset{ identifier < time > _ , identifier _* }` and "The operator can be
applied also Data Sets; the result Data Set has the same Identifier, Measure and Attribute
Components as the operand Data Set". The type names no time Measure, only the reference time
Identifier, so that is what moves: `Id_2` is shifted by one month, and `Me_1` is kept. (The
page does not say it in words; DS_1 has no other time component, so no other reading applies
to it.) `Id_2` is a *date*, so the result keeps the type of every component.

As the page's own `dateadd (2020-02-10, 1, “M”)` "gives 2020-03-10", adding a month keeps the
day of the month: 2019-01-15 → 2019-02-15 and 2020-02-10 → 2020-03-10.
