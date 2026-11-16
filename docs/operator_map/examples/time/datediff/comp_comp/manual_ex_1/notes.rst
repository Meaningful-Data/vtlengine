:Tests: ``datediff(Id_2, Me_1)`` inside ``calc``, from a date identifier to a time_period measure.
:Manual: `Days between two dates <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Days%20between%20two%20dates/index.html>`__, Example 1, output verbatim: "The operator datediff returns the number of days between two dates or time periods. The last day of the time period is assumed as the starting/ending date."
:Expected: ``2020Q2`` ends on 2020-06-30 and ``2021Q1`` on 2021-03-31, so ``Me_2`` is 546, 639 and 90, Integers.
