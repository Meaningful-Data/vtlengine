:Tests: the cross join of two operands with no common component, with no clause.
:Manual: `Cross Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__: it performs "a cartesian product of the input operands", and "The final result has a size equal to the product of the sizes of each input data set".
:Expected: the cartesian product of ``DS_1`` (``Id_1`` 1, 2) and ``DS_2`` (``Id_2`` A, B) has 2 × 2 = 4 data points, ``(1,A)``, ``(1,B)``, ``(2,A)`` and ``(2,B)``, with each component once, under its name and role.
