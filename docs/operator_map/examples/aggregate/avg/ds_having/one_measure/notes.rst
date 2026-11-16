:Tests: ``avg`` with the ``having`` of *Two measures*, on a single-measure operand: the shared input without ``Me_2``.
:Manual: `Aggregate invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html>`__: "only groups for which havingCondition evaluates to true appear in the result"; the page does not restrict ``having`` to single-measure operands.
:Expected: ``avg ( Me_1 ) > 3`` holds for ``B`` only, which is kept with ``Me_1`` = 6.
