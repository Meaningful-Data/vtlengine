:Tests: ``apply DS_1 + DS_2``, written with the data set names, on the homonym measure ``Me_1``.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__: apply "applies the expression to each of the n-uple of homonym Measures in the input data sets producing in the target a single homonym Measure for each n-uple", and "All the input measures do not appear".
:Expected: the natural join keeps ``(1,A)``, ``(2,A)`` and ``(2,B)``, the keys in both operands: ``Me_1 := DS_1#Me_1 + DS_2#Me_1`` is 11, 33 and 44.
