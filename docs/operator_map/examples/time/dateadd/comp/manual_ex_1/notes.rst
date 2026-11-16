:Tests: ``dateadd(Me_1, 2, "M")`` inside ``calc``: two months after each date.
:Manual: `Add time unit to date <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Add%20time%20unit%20to%20date/index.html>`__, Example 1, output verbatim: "adding months to a given date returns the date plus integer months", and "In the component version, that can be used in a calc clause, a new component of type date is added to the dataset."
:Expected: ``Me_2`` is 2019-03-01, 2019-09-01 and 2021-02-28, two months after 2020-12-31 being the last day of February 2021.
