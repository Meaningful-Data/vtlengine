:Tests: a natural inner join of three operands with the same identifiers.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: the contents "are ideally determined stepwise ... joining each of the other input data sets to the partial result".
:Expected: ``DS_1`` and ``DS_2`` match at ``(1,A)``, ``(2,A)`` and ``(2,B)``, and of these ``DS_3`` has ``(1,A)`` and ``(2,B)``: two data points, with ``Me_1``, ``Me_2`` and ``Me_3``.
