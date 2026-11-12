:Tests: ``nvl ( Me_1, Me_2 )`` inside ``calc``: two components of the same data set, with a null ``Me_1`` and, in one data point, a null ``Me_2`` as well.
:Manual: `Nvl <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Nvl/index.html>`__: "The operator **nvl** returns the value from *op2* when the value from *op1* is null, otherwise it returns the value from *op1*."
:Expected: ``Me_3`` is 10, 200 and null: ``Me_1`` is kept where it is not null; where it is null the result is ``Me_2``, which is null at id 3.
