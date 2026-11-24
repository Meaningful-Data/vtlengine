:Tests: ``cast(Me_1, date)`` on a Time measure whose intervals start and end on the same day, an explicit cast without mask.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **time** to **date**: A ``date`` value is returned if the ``time`` value start and ends at the same date, otherwise an error is raised."
:Expected: ``2020-01-15`` and ``2021-12-31``; neither raises the error.
