:Tests: ``calc measure Me_3 := Me_1 + Me_2``, with the role keyword written out.
:Manual: `Full Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html>`__: "calc calculates new components by independent sub-expressions evaluated at component level. The role of the new components can be specified".
:Expected: the full join keeps the five keys of both operands, with ``Me_2`` null at ``(1,B)`` and ``Me_1`` null at ``(1,C)``. The input components stay and ``Me_3`` is added: 11, null, null, 33 and 44, as a sum with a null is null.
