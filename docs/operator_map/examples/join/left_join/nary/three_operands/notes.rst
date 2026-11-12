:Tests: a left join of three operands with the same identifiers.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__: at each step "a data point in VDS₁ is generated for each datapoint in the partial result", and the measures of an operand that lacks the key are null.
:Expected: the four keys of ``DS_1``, with ``Me_2`` null at ``(1,B)`` and ``Me_3`` null at ``(1,B)`` and ``(2,A)``; ``(1,C)`` and ``(3,A)`` are not kept.
