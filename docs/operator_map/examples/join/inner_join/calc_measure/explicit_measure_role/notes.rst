:Tests: ``calc measure Me_3 := Me_1 + Me_2``, with the role keyword written out.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: "calc calculates new components by independent sub-expressions evaluated at component level. The role of the new components can be specified".
:Expected: the natural join keeps ``(1,A)``, ``(2,A)`` and ``(2,B)``, the keys in both operands. The input components stay and ``Me_3`` is added: 11, 33 and 44.
