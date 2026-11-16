:Tests: ``avg`` grouped by ``Id_1`` over a measure with one null, the ``Me_1`` of ``A, 2021Q1``.
:Manual: the `User Manual <https://sdmx-twg.github.io/vtl/v2.2/user_manual/operators.html>`__: "the input Data Points having a null value are in general not considered. In the average, for example, they are not considered both in the numerator (the sum) and in the denominator (the count)."
:Expected: (3 + 1 + 1) / 3 = 1.666667 for ``A``, where counting the null would give 1.25, and 6 for ``B``, as Numbers: "The result is always a number".
