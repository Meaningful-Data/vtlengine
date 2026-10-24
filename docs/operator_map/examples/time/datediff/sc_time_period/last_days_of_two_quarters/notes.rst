:Tests: ``datediff`` of two scalar time periods, the page's valid syntax ``datediff (2022Q1, 2023Q2)``.
:Manual: `Days between two dates <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Days%20between%20two%20dates/index.html>`__: "The last day of the time period is assumed as the starting/ending date."
:Expected: ``DS_r = 456``, an Integer: from 2022-03-31, the end of ``2022Q1``, to 2023-03-31 is 365 days, with no February 29 in between, then 30, 31 and 30 days to 2023-06-30, the end of ``2023Q2``.
