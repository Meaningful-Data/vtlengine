:Tests: ``max`` with ``having``, over the two measures of the shared input.
:Manual: `Aggregate invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html>`__: the havingCondition "must invoke aggregate operators", and "only groups for which havingCondition evaluates to true appear in the result". The condition aggregates ``Me_1`` only, where the manual and the reading of #1119 agree.
:Expected: ``max ( Me_1 ) < 6`` is 5 for ``A`` and 8 for ``B``, so only ``A`` is kept, with 5 and 3.5.
