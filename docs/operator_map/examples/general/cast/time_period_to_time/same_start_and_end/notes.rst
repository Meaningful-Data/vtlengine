:Tests: ``cast(Me_1, time)`` on a Time_Period measure, an implicit conversion written as an explicit cast, whose "behavior remains the same".
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **time_period** to **time**: a ``time`` value is returned having the stame start and end of the input ``time_period`` value."
:Expected: ``2020Q1`` gives ``2020-01-01/2020-03-31`` and ``2021M02`` gives ``2021-02-01/2021-02-28``, 2021 not being a leap year: a ``time`` written ``start/end``, as vtlengine reads it.
:Finding: pandas matches; DuckDB returns ``2020-Q1`` and ``2021-M02``.
