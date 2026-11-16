:Tests: ``nvl ( Me_1, 0 )`` inside ``calc``: a component and a scalar. The new measure ``Me_2`` is added and ``Me_1`` is left as it is.
:Manual: `Nvl <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Nvl/index.html>`__: "The operator **nvl** returns the value from *op2* when the value from *op1* is null, otherwise it returns the value from *op1*."
:Expected: ``Me_2`` is 10, 0 and 30: the null ``Me_1`` of id 2 becomes 0 in ``Me_2`` and stays null in ``Me_1``.
