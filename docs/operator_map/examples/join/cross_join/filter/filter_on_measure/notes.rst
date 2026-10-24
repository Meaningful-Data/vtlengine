:Tests: ``filter Me_1 > 15`` on the pairs of the product.
:Manual: `Cross Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__: the filter "is applied on VDS₁ ... when the expression is true the Data Point is kept in the result, otherwise it is discarded".
:Expected: the cartesian product of ``DS_1`` (``Id_1`` 1, 2) and ``DS_2`` (``Id_2`` A, B) has 2 × 2 = 4 data points; the two with ``Id_1`` = 2 (``Me_1`` = 20) are kept: ``(2,A)`` and ``(2,B)``.
