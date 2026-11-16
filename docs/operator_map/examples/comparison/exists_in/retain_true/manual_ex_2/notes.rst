:Tests: ``exists_in (DS_1, DS_2, true)``: only the data points of ``DS_1`` that are found in ``DS_2``.
:Manual: `Exists in <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Exists%20in/index.html>`__, Example 2, output verbatim: "If *retain* is **true** then only the data points with bool_var = **true** are returned."
:Expected: B, S, F and W, with ``bool_var`` true; G and M are left out, as ``DS_2`` has them only with ``Id_4`` = M.
