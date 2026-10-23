Manual: [Days between two dates](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Days%20between%20two%20dates/index.html), the valid syntax `datediff (2022Q1, 2023Q2)`.

"The last day of the time period is assumed as the starting/ending date": 2022Q1 ends on
2022-03-31 and 2023Q2 on 2023-06-30. From 2022-03-31 to 2023-03-31 is 365 days (no February 29
in between), then April 30, May 31 and June 30: 365 + 91 = 456.
