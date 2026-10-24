:Tests: ``calc Me_3 := Me_1 + Me_2`` with no role.
:Manual: `Full Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html>`__: in a join, "The role of the new components can be specified ...; if the role is not specified, it will be measure".
:Expected: the full join keeps the five keys of both operands, with ``Me_2`` null at ``(1,B)`` and ``Me_1`` null at ``(1,C)``. ``Me_3`` is a Measure, the result of Explicit measure role: 11, null, null, 33 and 44, as a sum with a null is null.
