:Tests: ``cast(Me_1, date)`` on a Time_Period measure of daily periods, an explicit cast without mask.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **time_period** to **date**: A ``date`` value is returned if the input ``time_period`` has a duration of a single day, representing that day; otherwise an error is raised."
:Expected: ``2020D15`` gives ``2020-01-15`` and ``2020D61`` gives ``2020-03-01``, 2020 being a leap year; both are daily, so neither raises the error.
