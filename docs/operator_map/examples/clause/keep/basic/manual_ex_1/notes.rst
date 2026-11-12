:Tests: ``keep Me_1``, on a data set with the measures ``Me_1`` and ``Me_2`` and the attribute ``At_1``.
:Manual: `Maintaining Components <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Maintaining%20Components/index.html>`__, Example 1, output verbatim: "The operator maintains the specified Components, drops all the other dependent Components of the Data Set (Measures and Attributes) and maintains the independent Components (Identifiers) unchanged."
:Expected: The three identifiers and ``Me_1``, on the same data points; ``Me_2`` and ``At_1`` are dropped.
