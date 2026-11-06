:Tests: ``cast(Me_1, string)`` on a Duration measure, without a mask: the default format.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **duration**: the ``duration`` value is printed using the ISO-8601 representation for durations."
:Expected: The inputs are the one-letter codes that vtlengine reads ("A Year", "M Month", "D Day" in the mask section), and one year, month and day are ``P1Y``, ``P1M`` and ``P1D`` in ISO-8601.
