:Tests: a cross join of three operands with no common component.
:Manual: `Cross Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__: "In each step, a data point in VDS₁ is generated for each pair of data points in the partial result and the joined data set", so "The final result has a size equal to the product of the sizes of each input data set".
:Expected: ``DS_3`` has ``Id_3`` P and Q: 2 × 2 × 2 = 8 data points, with ``Id_1``, ``Id_2``, ``Id_3`` and the three measures.
