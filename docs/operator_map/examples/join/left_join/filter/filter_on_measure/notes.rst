:Tests: ``filter Me_1 > 15`` on the joined data points.
:Manual: `Left Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__: the filter "is applied on VDS₁ ... when the expression is true the Data Point is kept in the result, otherwise it is discarded".
:Expected: ``(1,B)``, ``(2,A)`` and ``(2,B)`` (``Me_1`` = 20, 30 and 40), with ``Me_2`` null at ``(1,B)``; ``(1,A)`` has 10.
