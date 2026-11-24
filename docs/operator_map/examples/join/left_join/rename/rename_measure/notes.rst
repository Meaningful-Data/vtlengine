:Tests: ``rename Me_1 to Me_10``.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__: rename gives "each specified component with a new name. If the chosen name already exists in VDS₄, an error is raised."
:Expected: ``Me_10`` is not in VDS₄, so no error: the four keys of ``DS_1``, with ``Me_10`` (10, 20, 30, 40) and ``Me_2`` (1, null, 3, 4).
