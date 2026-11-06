:Tests: ``cast(Me_1, time_period)`` on a Date measure, an explicit cast without mask.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **date** to **time_period**: A ``time_period`` value with a daily duration is returned, starting and ending at the specified ``date`` value."
:Expected: Daily periods, written ``YYYYD`` and the day of the year: ``2020D15``, ``2020D61`` for March 1st, after the 31 days of January and the 29 of February, and ``2021D365``, 2021 not being a leap year.
