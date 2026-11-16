:Tests: ``dayofyear(Me_1)`` inside ``calc``, on March 1st of a common and of a leap year, and on the last day of a leap year.
:Manual: `Extract time period <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Extract%20time%20period/index.html>`__: "The operator **dayofyear** returns the ordinal day within the year (between 1 and 366)."
:Expected: ``Me_2`` is 60 (31 + 28 + 1, in 2019), 61 (31 + 29 + 1, in 2020) and 366, Integers.
