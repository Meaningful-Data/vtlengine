:Tests: a full join of three operands with the same identifiers.
:Manual: `Full Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html>`__: at each step "an additional data point is added to this set for each data point in the current join operand that wasn't matched before".
:Expected: the first step gives the five keys of ``DS_1`` and ``DS_2``, the second adds ``(3,A)`` of ``DS_3``: six data points, each measure null where its operand lacks the key.
