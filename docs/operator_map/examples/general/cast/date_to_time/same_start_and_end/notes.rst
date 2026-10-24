:Tests: ``cast(Me_1, time)`` on a Date measure, an implicit conversion written as an explicit cast, whose "behavior remains the same".
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **date** to **time**: a ``time`` value is returned having its start equal to its end, and both equal to the input ``date`` value."
:Expected: ``2020-01-15/2020-01-15`` and ``2021-12-31/2021-12-31``, a ``time`` written ``start/end``, as vtlengine reads it.
:Finding: pandas matches; DuckDB returns the date with a time of day, ``2020-01-15 00:00:00``.
