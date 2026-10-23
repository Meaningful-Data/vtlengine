Manual: [Days between two dates](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Days%20between%20two%20dates/index.html), the valid syntax `datediff (2020-12-14,2021-04-20)`
(VTL has no date literal: the dates are casts of strings, in the default `YYYY-MM-DD` form).

"The operator datediff returns the number of days between two dates": from 2020-12-14 to
2020-12-31 is 17 days, then January 31, February 28 (2021 is not a leap year), March 31 and 20
days of April: 17 + 31 + 28 + 31 + 20 = 127.
