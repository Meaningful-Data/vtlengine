:Tests: ``calc Me_3 := Me_1 + Me_2`` with no role.
:Manual: `Cross Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__: in a join, "The role of the new components can be specified ...; if the role is not specified, it will be measure".
:Expected: the cartesian product of ``DS_1`` (``Id_1`` 1, 2) and ``DS_2`` (``Id_2`` A, B) has 2 × 2 = 4 data points. ``Me_3`` is a Measure, as in Explicit measure role of the ``calc_measure`` form: 11, 12, 21 and 22.
