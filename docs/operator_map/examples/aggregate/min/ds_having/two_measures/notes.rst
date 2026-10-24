:Tests: ``min`` with ``having``, over the two measures of the shared input.
:Manual: `Aggregate invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html>`__: the havingCondition "must invoke aggregate operators", and "only groups for which havingCondition evaluates to true appear in the result". The condition aggregates ``Me_1`` only, where the manual and the reading of #1119 agree.
:Expected: ``min ( Me_1 ) < 2`` is 1 for ``A`` and 4 for ``B``, so only ``A`` is kept, with 1 and 0.5.
:Finding: both engines refuse it with SemanticError ``1-1-1-4``, "Operation not allowed for multimeasure Datasets"; *One measure* passes.
