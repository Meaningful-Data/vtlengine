:Tests: ``var_pop`` with ``having``, over the two measures of the shared input.
:Manual: `Aggregate invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html>`__: the havingCondition "must invoke aggregate operators", and "only groups for which havingCondition evaluates to true appear in the result". The condition aggregates ``Me_1`` only, where the manual and the reading of #1119 agree.
:Expected: ``var_pop ( Me_1 ) < 3`` is 2.75 for ``A`` and 4 for ``B``, so only ``A`` is kept, with 2.75 and 1.25.
