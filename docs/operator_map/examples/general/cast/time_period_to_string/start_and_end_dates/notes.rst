:Tests: ``cast(Me_1, string)`` on a Time_Period measure, without a mask: the default format.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **time_period**: the ``time_period`` value is printed using the ISO-8601 representation for generic time intervals, up to a precision of days, in the form start/end."
:Expected: the period codes: ``2020Q1`` gives ``2020Q1`` and ``2021M02`` gives ``2021M2``; a string is compared exactly.
:Team decision: `#1254 <https://github.com/Meaningful-Data/vtlengine/issues/1254>`__: the text of ``cast`` follows the pandas engine, which writes the period code, where the manual prints the interval ``start/end`` (``2020-01-01/2020-03-31``).
