:Tests: ``calc viral attribute At_1 := "EP"``, a constant viral attribute.
:Manual: `Cross Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__: on viral attributes, "if the calc clause is present instead, it may directly replace the computation algorithm used for the propagation".
:Expected: the cartesian product of ``DS_1`` (``Id_1`` 1, 2) and ``DS_2`` (``Id_2`` A, B) has 2 × 2 = 4 data points, and each gets the String Viral Attribute ``At_1`` = ``EP``.
