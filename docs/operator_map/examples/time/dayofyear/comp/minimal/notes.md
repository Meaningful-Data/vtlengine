Manual: [Extract time period](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Extract%20time%20period/index.html): "The operator **dayofyear** returns the ordinal day
within the year (between 1 and 366)", in a calc as a new integer component:

- 2019-03-01: 31 (January) + 28 (February, 2019 is not a leap year) + 1 = 60;
- 2020-03-01: 31 + 29 (2020 is a leap year) + 1 = 61;
- 2020-12-31: the last day of a leap year, 366.
