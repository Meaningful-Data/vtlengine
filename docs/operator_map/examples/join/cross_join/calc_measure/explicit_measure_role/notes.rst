:Tests: ``calc measure Me_3 := Me_1 + Me_2``, with the role keyword written out.
:Manual: `Cross Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__: "calc calculates new components by independent sub-expressions evaluated at component level. The role of the new components can be specified".
:Expected: the cartesian product of ``DS_1`` (``Id_1`` 1, 2) and ``DS_2`` (``Id_2`` A, B) has 2 × 2 = 4 data points. The input components stay and ``Me_3`` is added: 11, 12, 21 and 22.
