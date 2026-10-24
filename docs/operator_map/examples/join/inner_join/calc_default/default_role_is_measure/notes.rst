:Tests: ``calc Me_3 := Me_1 + Me_2`` with no role.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: in a join, "The role of the new components can be specified ...; if the role is not specified, it will be measure".
:Expected: the natural join keeps ``(1,A)``, ``(2,A)`` and ``(2,B)``, the keys in both operands. ``Me_3`` is a Measure, the result of Explicit measure role: 11, 33 and 44.
