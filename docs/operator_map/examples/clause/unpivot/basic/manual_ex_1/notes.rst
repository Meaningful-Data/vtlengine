:Tests: ``unpivot Id_2, Me_1``, on the Integer measures ``A``, ``B`` and ``C``.
:Manual: `Unpivoting <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Unpivoting/index.html>`__, Example 1, output verbatim: "The VDS identifier is assigned a value equal to the name of the Measure M of op", and "The VDS measure is assigned a value equal to the value of the Measure M of op".
:Expected: Six data points, one per data point and measure: ``Id_2`` holds ``A``, ``B`` or ``C`` and ``Me_1`` its value, such as ``(1, A)`` = 5 and ``(2, C)`` = 9.
