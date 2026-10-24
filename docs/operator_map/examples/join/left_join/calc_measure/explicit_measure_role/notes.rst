:Tests: ``calc measure Me_3 := Me_1 + Me_2``, with the role keyword written out.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__: "calc calculates new components by independent sub-expressions evaluated at component level. The role of the new components can be specified".
:Expected: the left join keeps the four keys of ``DS_1``, with ``Me_2`` null at ``(1,B)``. The input components stay and ``Me_3`` is added: 11, null, 33 and 44, as a sum with a null is null.
