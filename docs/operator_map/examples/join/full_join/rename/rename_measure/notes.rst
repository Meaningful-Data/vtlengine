:Tests: ``rename Me_1 to Me_10``.
:Manual: `Full Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html>`__: rename gives "each specified component with a new name. If the chosen name already exists in VDS₄, an error is raised."
:Expected: ``Me_10`` is not in VDS₄, so no error: the five keys, with ``Me_10`` (10, 20, null, 30, 40) and ``Me_2`` (1, null, 2, 3, 4).
