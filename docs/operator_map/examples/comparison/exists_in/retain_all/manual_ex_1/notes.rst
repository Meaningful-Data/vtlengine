:Tests: ``exists_in (DS_1, DS_2, all)``: every data point of ``DS_1`` is returned, found in ``DS_2`` or not.
:Manual: `Exists in <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Exists%20in/index.html>`__, Example 1, output verbatim: "If *retain* is **all** then both the Data Points having bool_var = **true** and bool_var = **false** are returned."
:Expected: ``bool_var`` is true for B, S, F and W, and false for G and M, which ``DS_2`` has only with ``Id_4`` = M, where ``DS_1`` has Total; all six data points are returned.
