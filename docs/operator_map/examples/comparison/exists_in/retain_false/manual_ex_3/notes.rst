:Tests: ``exists_in (DS_1, DS_2, false)``: only the data points of ``DS_1`` that are not found in ``DS_2``.
:Manual: `Exists in <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Exists%20in/index.html>`__, Example 3, output verbatim: "If *retain* is **false** then only the Data Points with bool_var = **false** are returned."
:Expected: G and M, with ``bool_var`` false, as ``DS_2`` has them only with ``Id_4`` = M; B, S, F and W are left out.
