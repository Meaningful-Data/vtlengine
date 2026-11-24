:Tests: ``filter Me_1 > 15`` on the joined data points.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: the filter "is applied on VDS₁ ... when the expression is true the Data Point is kept in the result, otherwise it is discarded".
:Expected: of the joined keys ``(1,A)``, ``(2,A)`` and ``(2,B)``, the last two (``Me_1`` = 30 and 40) are kept.
