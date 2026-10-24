:Tests: ``rank`` with ``partition except all``, ordered by ``Id_1``, which the script writes ``id_1``.
:Manual: `Analytic invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Analytic%20invocation/index.html>`__, Example 2, with the erratum below: ``partition except all`` leaves no identifier to partition by, and the `User Manual <https://sdmx-twg.github.io/vtl/v2.2/user_manual/transformations.html>`__ makes regular names "case insensitive", so ``id_1`` is ``Id_1``.
:Expected: one partition ordered by ``Id_1``: the four 2010 data points tie at rank 1, the four 2011 ones at 5 and the two 2012 ones at 9.
:Finding: both engines refuse it with SemanticError ``1-1-1-10``, "Component id_1 not found in Dataset DS_1".
:Erratum: the manual declares ``Me_1`` and ``Me_2`` Number, but ``calc`` keeps ``Me_1`` Integer as in ``DS_1``, and ``rank`` returns integer, so both are Integer here.
