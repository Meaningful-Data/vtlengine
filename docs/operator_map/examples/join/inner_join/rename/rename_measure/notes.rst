:Tests: ``rename Me_1 to Me_10``.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: rename gives "each specified component with a new name. If the chosen name already exists in VDS₄, an error is raised."
:Expected: ``Me_10`` is not in VDS₄, so no error: the natural join keeps ``(1,A)``, ``(2,A)`` and ``(2,B)``, the keys in both operands, with ``Me_10`` (10, 30, 40) and ``Me_2``.
