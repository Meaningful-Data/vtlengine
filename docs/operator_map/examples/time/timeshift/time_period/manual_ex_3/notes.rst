:Tests: ``timeshift(DS_3, 1)`` on annual time_period series: one period into the future.
:Manual: `Time shift <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20shift/index.html>`__, Example 3, output verbatim: the operator "shifts the reference time Identifier of a number of periods (of the time series) equal to the *shiftNumber* parameter. If *shiftNumber* is negative, the shift is in the past, otherwise it is in the future."
:Expected: Each year moves one year on, 2010 to 2013 becoming 2011 to 2014, with the measures as they were.
