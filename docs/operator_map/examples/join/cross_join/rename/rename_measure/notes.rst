:Tests: ``rename Me_1 to Me_10``.
:Manual: `Cross Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__: rename gives "each specified component with a new name. If the chosen name already exists in VDS₄, an error is raised."
:Expected: ``Me_10`` is not in VDS₄, so no error: the cartesian product of ``DS_1`` (``Id_1`` 1, 2) and ``DS_2`` (``Id_2`` A, B) has 2 × 2 = 4 data points, with ``Me_10`` (10, 10, 20, 20) and ``Me_2``.
