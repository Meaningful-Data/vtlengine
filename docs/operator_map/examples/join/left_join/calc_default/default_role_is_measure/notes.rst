:Tests: ``calc Me_3 := Me_1 + Me_2`` with no role.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__: in a join, "The role of the new components can be specified ...; if the role is not specified, it will be measure".
:Expected: the left join keeps the four keys of ``DS_1``, with ``Me_2`` null at ``(1,B)``. ``Me_3`` is a Measure, as in Explicit measure role of the ``calc_measure`` form: 11, null, 33 and 44, as a sum with a null is null.
