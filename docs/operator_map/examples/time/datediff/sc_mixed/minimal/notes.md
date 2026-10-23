Manual: [Days between two dates](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Days%20between%20two%20dates/index.html), the valid syntax `datediff (2021Q2, 2021-11-04)`,
whose result the manual gives in an example misplaced on the
[Extract time period](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Extract%20time%20period/index.html) page (its Example 2): `datediff (2021Q2, 2021-11-04);`
"returns 127".

"The last day of the time period is assumed as the starting/ending date": 2021Q2 ends on
2021-06-30. From there to 2021-11-04: July 31, August 31, September 30, October 31 and 4 days of
November: 31 + 31 + 30 + 31 + 4 = 127.
